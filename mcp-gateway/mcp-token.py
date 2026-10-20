#!/usr/bin/env python3
"""Manage mcp-gateway bearer tokens (host-side, stdlib only).

  mcp-token.py add <name> [--servers musicgen,imagegen]   mint a token for one client
  mcp-token.py revoke <name>
  mcp-token.py list

Only the token's SHA-256 goes into tokens.json (read live by the gateway, no restart).
The token itself is written to <SECRETS_DIR>/<name>.token (mode 600) and never printed,
so it stays out of terminal scrollback and chat logs: open that file to copy it into
the client, then delete it if you like (a lost token is revoked and re-minted, not recovered).
"""

import argparse
import hashlib
import json
import os
import secrets
import sys
from datetime import date
from pathlib import Path

SECRETS_DIR = Path(os.environ.get("MCP_GATEWAY_SECRETS", "/run/secrets/mcp-gateway"))
# store/ (hashes only) is what the gateway mounts; the plaintext .token files stay outside it.
TOKENS_FILE = SECRETS_DIR / "store" / "tokens.json"
ALL_SERVERS = ["musicgen", "imagegen"]


def load() -> dict:
    return json.loads(TOKENS_FILE.read_text()) if TOKENS_FILE.exists() else {"tokens": []}


def save(data: dict) -> None:
    TOKENS_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    tmp = TOKENS_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.chmod(0o600)
    tmp.replace(TOKENS_FILE)  # atomic, so the gateway never reads a half-written file


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add")
    a.add_argument("name")
    a.add_argument("--servers", default=",".join(ALL_SERVERS))
    sub.add_parser("revoke").add_argument("name")
    sub.add_parser("list")
    args = p.parse_args()
    data = load()

    if args.cmd == "list":
        for t in data["tokens"]:
            print(f"{t['name']:20} {','.join(t['servers']):20} created {t.get('created', '?')}")
        return

    if args.cmd == "revoke":
        before = len(data["tokens"])
        data["tokens"] = [t for t in data["tokens"] if t["name"] != args.name]
        if len(data["tokens"]) == before:
            sys.exit(f"no token named {args.name}")
        save(data)
        (SECRETS_DIR / f"{args.name}.token").unlink(missing_ok=True)
        print(f"revoked {args.name}")
        return

    servers = [s.strip() for s in args.servers.split(",") if s.strip()]
    unknown = set(servers) - set(ALL_SERVERS)
    if unknown:
        sys.exit(f"unknown server(s): {', '.join(sorted(unknown))}")
    if any(t["name"] == args.name for t in data["tokens"]):
        sys.exit(f"{args.name} already exists; revoke it first to rotate")
    token = secrets.token_urlsafe(32)
    data["tokens"].append({
        "name": args.name,
        "sha256": hashlib.sha256(token.encode()).hexdigest(),
        "servers": servers,
        "created": date.today().isoformat(),
    })
    save(data)
    token_file = SECRETS_DIR / f"{args.name}.token"
    token_file.write_text(token + "\n")
    token_file.chmod(0o600)
    print(f"added {args.name} for {', '.join(servers)}; token written to {token_file}")


if __name__ == "__main__":
    main()
