#!/usr/bin/env python3
"""Minimal MCP client for testing the stack's servers from the host (stdlib only).

  mcpcall.py SERVER LIST                       list tools and their parameters
  mcpcall.py SERVER TOOL ['{"arg": 1}']        call one tool, print its result
  mcpcall.py --remote NAME SERVER ...          same, through $MCP_PUBLIC_URL like a
                                               remote client, reading the bearer token from
                                               $MCP_GATEWAY_SECRETS/NAME.token

SERVER is bible-api | imagegen | stack-api | musicgen (remote: musicgen | imagegen only).
Local calls block until the tool finishes; remote calls get the gateway's 40s job behavior.
The token is read from its file and never printed. Create a test token with
`mcp-gateway/mcp-token.py add local-test` and revoke it afterwards.
"""
import json
import os
import socket
import sys
import time
import urllib.parse
import urllib.request

LOCAL = {
    "bible-api": int(os.environ.get("BIBLE_API_PORT", "8001")),
    "imagegen": int(os.environ.get("IMAGE_API_PORT", "8010")),
    "stack-api": int(os.environ.get("STACK_API_PORT", "8020")),
    "musicgen": int(os.environ.get("MUSIC_API_PORT", "8030")),
}
PUBLIC_URL = os.environ.get("MCP_PUBLIC_URL", "https://mcp.example.com").rstrip("/")
# Optional DNS override for MCP_PUBLIC_URL's host. Leave unset to use normal DNS.
PUBLIC_IP = os.environ.get("MCP_PUBLIC_IP", "").strip()
SECRETS_DIR = os.environ.get("MCP_GATEWAY_SECRETS", "/run/secrets/mcp-gateway")

args = sys.argv[1:]
token = None
if args[:1] == ["--remote"]:
    name = args[1]
    token = open(os.path.join(SECRETS_DIR, f"{name}.token")).read().strip()
    args = args[2:]
if len(args) < 2:
    sys.exit(__doc__)
server, tool = args[0], args[1]
tool_args = json.loads(args[2]) if len(args) > 2 else {}

if token:
    url = f"{PUBLIC_URL}/{server}/mcp"
    if PUBLIC_IP:
        host = urllib.parse.urlparse(PUBLIC_URL).hostname
        _orig = socket.getaddrinfo

        def _resolve(h, *a, **k):
            return _orig(PUBLIC_IP if h == host else h, *a, **k)

        socket.getaddrinfo = _resolve
else:
    url = f"http://127.0.0.1:{LOCAL[server]}/mcp"

headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
if token:
    headers["Authorization"] = f"Bearer {token}"


def rpc(body, session=None):
    h = dict(headers, **({"mcp-session-id": session} if session else {}))
    req = urllib.request.Request(url, json.dumps(body).encode(), h)
    with urllib.request.urlopen(req, timeout=3600) as r:
        text = r.read().decode()
        data = [line[5:].strip() for line in text.splitlines() if line.startswith("data:")]
        return r.headers.get("mcp-session-id"), (json.loads(data[-1]) if data else None)


session, _ = rpc({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {
    "protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "mcpcall", "version": "1"}}})
rpc({"jsonrpc": "2.0", "method": "notifications/initialized"}, session)
started = time.monotonic()
if tool == "LIST":
    _, res = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/list"}, session)
    for t in res["result"]["tools"]:
        print(f"{t['name']}({', '.join(t['inputSchema'].get('properties', {}))})")
else:
    _, res = rpc({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                  "params": {"name": tool, "arguments": tool_args}}, session)
    result = res.get("result", res)
    if result.get("isError"):
        print("TOOL ERROR:", " ".join(c.get("text", "") for c in result.get("content", [])))
    else:
        print(json.dumps(result.get("structuredContent") or result, indent=1))
sys.stdout.flush()
print(f"[{time.monotonic() - started:.1f}s]", file=sys.stderr)
