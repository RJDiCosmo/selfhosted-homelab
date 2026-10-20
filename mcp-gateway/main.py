"""MCP gateway — bearer-token auth in front of the musicgen/imagegen MCP servers for remote
clients. Publish it through a reverse proxy at /<server>/mcp; the MCP servers themselves stay
unauthenticated and reachable only on the internal Docker network and localhost.

Tokens live hashed in TOKENS_FILE (managed by mcp-token.py on the host) and are re-read
whenever the file changes, so adding/revoking a token needs no restart.
"""

import hashlib
import hmac
import json
import logging
import os
import time
from collections import deque
from pathlib import Path

import httpx
import uvicorn
from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

UPSTREAMS = {
    "musicgen": "http://music-api:8030/mcp",
    "imagegen": "http://image-api:8010/mcp",
}
TOKENS_FILE = Path(os.environ.get("TOKENS_FILE", "/secrets/tokens.json"))
MAX_BODY = int(os.environ.get("MAX_BODY_BYTES", str(1024 * 1024)))
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "60"))
# Generations can take several minutes (XL cold start + 3-minute track), so reads are long.
UPSTREAM_TIMEOUT = httpx.Timeout(connect=10.0, read=900.0, write=60.0, pool=10.0)

# Never forwarded in either direction: hop-by-hop headers, the client's token, and
# headers the proxy recomputes.
DROP_HEADERS = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization", "te",
    "trailer", "transfer-encoding", "upgrade", "host", "authorization", "content-length",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("mcp-gateway")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one line per request is ours, not httpx's


class TokenStore:
    """{"tokens": [{"name": ..., "sha256": <hex of token>, "servers": [...]}]}"""

    def __init__(self, path: Path):
        self.path = path
        self.mtime: float | None = None
        self.tokens: list[dict] = []

    def _reload(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except FileNotFoundError:
            self.tokens, self.mtime = [], None
            return
        if mtime != self.mtime:
            self.tokens = json.loads(self.path.read_text()).get("tokens", [])
            self.mtime = mtime
            log.info("loaded %d token(s)", len(self.tokens))

    def lookup(self, token: str) -> dict | None:
        self._reload()
        digest = hashlib.sha256(token.encode()).hexdigest()
        match = None
        for entry in self.tokens:  # check every entry so timing doesn't reveal position
            if hmac.compare_digest(digest, entry.get("sha256", "")):
                match = entry
        return match


store = TokenStore(TOKENS_FILE)
recent: dict[str, deque] = {}
client = httpx.AsyncClient(timeout=UPSTREAM_TIMEOUT)


def _rate_limited(name: str) -> bool:
    now = time.monotonic()
    q = recent.setdefault(name, deque())
    while q and now - q[0] > 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        return True
    q.append(now)
    return False


def _rpc_label(body: bytes) -> str:
    """'tools/call generate_song' etc., for the log; '' for non-JSON-RPC bodies."""
    try:
        msg = json.loads(body)
    except ValueError:
        return ""
    msgs = msg if isinstance(msg, list) else [msg]
    labels = []
    for m in msgs:
        if isinstance(m, dict) and "method" in m:
            name = (m.get("params") or {}).get("name", "") if m["method"] == "tools/call" else ""
            labels.append(f"{m['method']} {name}".strip())
    return ",".join(labels)


def _unauthorized(error: str, status: int = 401) -> JSONResponse:
    return JSONResponse(
        {"error": error}, status_code=status,
        headers={"WWW-Authenticate": f'Bearer realm="mcp", error="{error}"'},
    )


async def proxy(request: Request):
    server = request.path_params["server"]
    upstream = UPSTREAMS.get(server)
    if upstream is None:
        return JSONResponse({"error": "not_found"}, status_code=404)

    scheme, _, token = request.headers.get("authorization", "").partition(" ")
    entry = store.lookup(token.strip()) if scheme.lower() == "bearer" and token.strip() else None
    peer = request.client.host if request.client else "?"
    if entry is None:
        log.info("DENY %s %s %s: bad or missing token", peer, request.method, server)
        return _unauthorized("invalid_token")
    name = entry.get("name", "?")
    if server not in entry.get("servers", []):
        log.info("DENY %s %s %s: token %s not allowed on this server", peer, request.method, server, name)
        return _unauthorized("insufficient_scope", 403)
    if _rate_limited(name):
        log.info("LIMIT %s %s %s", name, peer, server)
        return JSONResponse({"error": "rate_limited"}, status_code=429, headers={"Retry-After": "60"})

    if int(request.headers.get("content-length") or 0) > MAX_BODY:
        return JSONResponse({"error": "body_too_large"}, status_code=413)
    body = b""
    async for chunk in request.stream():
        body += chunk
        if len(body) > MAX_BODY:
            return JSONResponse({"error": "body_too_large"}, status_code=413)

    headers = {k: v for k, v in request.headers.items() if k.lower() not in DROP_HEADERS}
    # Tells the MCP servers this is a remote client (60s tool-call limit in Grok Bot), so
    # long tools hand back a job_id after ~40s instead of blocking. Always overwritten.
    headers = {k: v for k, v in headers.items() if k.lower() != "x-mcp-gateway"}
    headers["x-mcp-gateway"] = name
    upstream_req = client.build_request(
        request.method, upstream, params=request.query_params, headers=headers, content=body,
    )
    try:
        resp = await client.send(upstream_req, stream=True)
    except httpx.HTTPError as e:
        log.info("ERROR %s %s %s %s: upstream %s", name, peer, server, _rpc_label(body), type(e).__name__)
        return JSONResponse({"error": "upstream_unavailable"}, status_code=502)

    log.info("OK %s %s %s %s %s -> %d", name, peer, request.method, server, _rpc_label(body), resp.status_code)
    out_headers = {k: v for k, v in resp.headers.items() if k.lower() not in DROP_HEADERS}
    return StreamingResponse(
        resp.aiter_raw(), status_code=resp.status_code, headers=out_headers,
        background=BackgroundTask(resp.aclose),
    )


async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


app = Starlette(routes=[
    Route("/health", health, methods=["GET"]),
    Route("/{server}/mcp", proxy, methods=["GET", "POST", "DELETE"]),
])

if __name__ == "__main__":
    # Only the reverse proxy can reach this container, so its X-Forwarded-For is trusted for logging.
    uvicorn.run(app, host="0.0.0.0", port=8040, proxy_headers=True, forwarded_allow_ips="*",
                access_log=False)
