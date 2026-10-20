"""Tools for verse, passage, and chapter retrieval."""

from database import get_db, resolve_book
from mcp_server import mcp
from models import Verse, VerseList


async def _book_num_or_404(book: str) -> tuple[int, str]:
    """Return (book_num, book_name) or raise ValueError."""
    book_num = resolve_book(book)
    if book_num:
        async with get_db() as db:
            async with db.execute("SELECT name FROM books WHERE num=?", (book_num,)) as cur:
                row = await cur.fetchone()
                return book_num, row["name"]

    # Fallback: query DB directly
    async with get_db() as db:
        q = "SELECT num, name FROM books WHERE LOWER(name)=LOWER(?) OR LOWER(abbrev)=LOWER(?)"
        async with db.execute(q, (book, book)) as cur:
            row = await cur.fetchone()
        if not row:
            raise ValueError(f"Book not found: {book!r}")
        return row["num"], row["name"]


def _make_verse(row, book_name: str) -> Verse:
    trans = row["translation"].upper()
    ref = f"{book_name} {row['chapter']}:{row['verse']} ({trans})"
    return Verse(
        translation=row["translation"],
        book_num=row["book_num"],
        book_name=book_name,
        chapter=row["chapter"],
        verse=row["verse"],
        text=row["text"],
        reference=ref,
    )


@mcp.tool(name="verse")
async def get_verse(translation: str, book: str, chapter: int, verse: int) -> Verse:
    """Fetch a single verse from a given translation."""
    book_num, book_name = await _book_num_or_404(book)
    async with get_db() as db:
        async with db.execute(
            "SELECT translation, book_num, chapter, verse, text "
            "FROM verses WHERE translation=? AND book_num=? AND chapter=? AND verse=?",
            (translation.lower(), book_num, chapter, verse),
        ) as cur:
            row = await cur.fetchone()
    if not row:
        raise ValueError(
            f"{book_name} {chapter}:{verse} not found in translation '{translation}'. "
            "Check list_translations for available translations."
        )
    return _make_verse(row, book_name)


@mcp.tool(name="passage")
async def get_passage(
    translation: str, book: str, chapter: int, verse_start: int, verse_end: int
) -> VerseList:
    """Fetch a range of verses (verse_start..verse_end) within one chapter."""
    if verse_end < verse_start:
        raise ValueError("verse_end must be >= verse_start")
    book_num, book_name = await _book_num_or_404(book)
    async with get_db() as db:
        async with db.execute(
            "SELECT translation, book_num, chapter, verse, text "
            "FROM verses "
            "WHERE translation=? AND book_num=? AND chapter=? AND verse BETWEEN ? AND ? "
            "ORDER BY verse",
            (translation.lower(), book_num, chapter, verse_start, verse_end),
        ) as cur:
            rows = await cur.fetchall()
    if not rows:
        raise ValueError(
            f"{book_name} {chapter}:{verse_start}-{verse_end} not found in '{translation}'."
        )
    return VerseList(verses=[_make_verse(r, book_name) for r in rows])


@mcp.tool(name="chapter")
async def get_chapter(translation: str, book: str, chapter: int) -> VerseList:
    """Fetch every verse in a chapter."""
    book_num, book_name = await _book_num_or_404(book)
    async with get_db() as db:
        async with db.execute(
            "SELECT translation, book_num, chapter, verse, text "
            "FROM verses WHERE translation=? AND book_num=? AND chapter=? ORDER BY verse",
            (translation.lower(), book_num, chapter),
        ) as cur:
            rows = await cur.fetchall()
    if not rows:
        raise ValueError(f"{book_name} chapter {chapter} not found in '{translation}'.")
    return VerseList(verses=[_make_verse(r, book_name) for r in rows])


@mcp.tool(name="compare")
async def compare_verse(
    book: str,
    chapter: int,
    verse: int,
    translations: str = "kjv,asv,nheb",
) -> VerseList:
    """Return the same verse in multiple translations for comparison (comma-separated translation IDs)."""
    book_num, book_name = await _book_num_or_404(book)
    trans_list = [t.strip().lower() for t in translations.split(",") if t.strip()]
    placeholders = ",".join("?" * len(trans_list))

    async with get_db() as db:
        async with db.execute(
            f"SELECT translation, book_num, chapter, verse, text "
            f"FROM verses "
            f"WHERE translation IN ({placeholders}) "
            f"AND book_num=? AND chapter=? AND verse=? ORDER BY translation",
            (*trans_list, book_num, chapter, verse),
        ) as cur:
            rows = await cur.fetchall()

    return VerseList(verses=[_make_verse(r, book_name) for r in rows])
