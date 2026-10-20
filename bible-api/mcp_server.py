"""Shared FastMCP instance — routers import `mcp` from here to register tools."""

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="Bible API",
    instructions=(
        "Local Bible database serving multiple translations, Strong's concordance, "
        "and original Hebrew/Greek/Aramaic texts with word-level interlinear data."
    ),
    host="0.0.0.0",
    port=8000,
)
