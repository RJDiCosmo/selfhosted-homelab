"""Bible API — MCP server entry point."""

import asyncio

from starlette.requests import Request
from starlette.responses import JSONResponse

from database import startup
from mcp_server import mcp

# Importing these registers their @mcp.tool() functions on the shared `mcp` instance.
from routers import books, lexicon, search, verses  # noqa: F401


@mcp.custom_route("/health", methods=["GET"])
async def health(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


if __name__ == "__main__":
    asyncio.run(startup())
    mcp.run(transport="streamable-http")
