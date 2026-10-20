"""Tools for translations and books metadata."""

from database import get_db
from mcp_server import mcp
from models import Book, BookList, Translation, TranslationList


@mcp.tool(name="translations")
async def list_translations() -> TranslationList:
    """List all loaded Bible translations."""
    async with get_db() as db:
        async with db.execute(
            "SELECT id, name, language, direction FROM translations ORDER BY language, name"
        ) as cur:
            rows = [Translation(**dict(row)) for row in await cur.fetchall()]
    return TranslationList(translations=rows)


@mcp.tool(name="books")
async def list_books(testament: str | None = None) -> BookList:
    """List all Bible books, optionally filtered to 'OT' or 'NT'."""
    sql = "SELECT num, name, abbrev, testament, chapters FROM books"
    params: list = []
    if testament:
        sql += " WHERE testament=?"
        params.append(testament.upper())
    sql += " ORDER BY num"

    async with get_db() as db:
        async with db.execute(sql, params) as cur:
            rows = [Book(**dict(row)) for row in await cur.fetchall()]
    return BookList(books=rows)


@mcp.tool(name="book")
async def get_book(book: str) -> Book:
    """Look up a single book by number, name, or abbreviation."""
    async with get_db() as db:
        # Try by number first
        row = None
        if book.isdigit():
            async with db.execute(
                "SELECT num, name, abbrev, testament, chapters FROM books WHERE num=?",
                (int(book),),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            async with db.execute(
                "SELECT num, name, abbrev, testament, chapters FROM books "
                "WHERE LOWER(name)=LOWER(?) OR LOWER(abbrev)=LOWER(?)",
                (book, book),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            raise ValueError(f"Book not found: {book!r}")
        return Book(**dict(row))
