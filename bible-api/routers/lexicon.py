"""Strong's concordance and interlinear tools."""

import re

from database import get_db, resolve_book
from mcp_server import mcp
from models import (
    CrossReference,
    InterlinearVerse,
    InterlinearWord,
    LexiconEntry,
    OccurrenceList,
)


def _normalise_strongs(raw: str) -> str:
    """Convert user input like 'H1', 'H0001', 'g26', 'G026' → 'H0001' / 'G0026'."""
    raw = raw.strip().upper()
    m = re.match(r"([HGA])(\d+)([A-Z]?)", raw)
    if not m:
        raise ValueError(f"Invalid Strong's number: {raw!r}. Expected H#### or G####.")
    prefix, digits, suffix = m.groups()
    return f"{prefix}{int(digits):04d}"


@mcp.tool(name="strongs")
async def get_strongs(strongs_num: str) -> LexiconEntry:
    """Look up a Strong's concordance entry, e.g. 'H0430' or 'G0026'."""
    num = _normalise_strongs(strongs_num)

    async with get_db() as db:
        async with db.execute(
            "SELECT strongs_num, language, word, transliteration, pronunciation, "
            "definition, kjv_usage FROM lexicon WHERE strongs_num=?",
            (num,),
        ) as cur:
            row = await cur.fetchone()

    if not row:
        raise ValueError(f"Strong's {num} not found. "
                          "Check that the lexicon import (03_import_strongs.py) has been run.")
    return LexiconEntry(**dict(row))


@mcp.tool(name="strongs_verses")
async def strongs_occurrences(
    strongs_num: str,
    translation: str = "kjv",
    limit: int = 20,
) -> OccurrenceList:
    """Return verses where a Strong's number appears, via word_occurrences."""
    num = _normalise_strongs(strongs_num)

    async with get_db() as db:
        async with db.execute(
            """SELECT v.translation, b.name AS book_name, v.chapter, v.verse, v.text
               FROM word_occurrences wo
               JOIN verses v ON v.id = wo.verse_id
               JOIN books  b ON b.num = v.book_num
               WHERE wo.strongs_num=? AND v.translation=?
               ORDER BY v.book_num, v.chapter, v.verse
               LIMIT ?""",
            (num, translation.lower(), limit),
        ) as cur:
            rows = await cur.fetchall()

    return OccurrenceList(occurrences=[
        {
            "reference": f"{r['book_name']} {r['chapter']}:{r['verse']} ({r['translation'].upper()})",
            "text": r["text"],
        }
        for r in rows
    ])


async def _interlinear_for_verse(
    db, book_num: int, chapter: int, verse: int, orig_translation: str
) -> list[InterlinearWord]:
    """Fetch word_occurrences + lexicon for an original-language verse."""
    async with db.execute(
        "SELECT id, text FROM verses "
        "WHERE translation=? AND book_num=? AND chapter=? AND verse=?",
        (orig_translation, book_num, chapter, verse),
    ) as cur:
        verse_row = await cur.fetchone()
    if not verse_row:
        return []

    verse_id = verse_row["id"]
    orig_words = verse_row["text"].split()

    async with db.execute(
        """SELECT wo.position, wo.strongs_num, wo.morphology,
                  l.word, l.transliteration, l.definition
           FROM word_occurrences wo
           LEFT JOIN lexicon l ON l.strongs_num = wo.strongs_num
           WHERE wo.verse_id=?
           ORDER BY wo.position""",
        (verse_id,),
    ) as cur:
        occ_rows = await cur.fetchall()

    words: list[InterlinearWord] = []
    for row in occ_rows:
        pos = row["position"]
        orig_word = orig_words[pos] if pos < len(orig_words) else row["word"] or ""
        words.append(InterlinearWord(
            position=pos,
            word=orig_word,
            strongs_num=row["strongs_num"],
            transliteration=row["transliteration"],
            definition=row["definition"],
            morphology=row["morphology"],
        ))

    # If no word_occurrences (import may not have morph data), fall back to bare words
    if not words:
        words = [
            InterlinearWord(position=i, word=w, strongs_num=None,
                            transliteration=None, definition=None, morphology=None)
            for i, w in enumerate(orig_words)
        ]

    return words


@mcp.tool(name="interlinear")
async def get_interlinear(book: str, chapter: int, verse: int) -> InterlinearVerse:
    """
    Return original-language interlinear data for any verse.
    OT verses use WLC (Hebrew), NT verses use SBLGNT (Greek).
    """
    book_num = resolve_book(book)
    if not book_num:
        async with get_db() as db:
            async with db.execute(
                "SELECT num FROM books WHERE LOWER(name)=LOWER(?) OR LOWER(abbrev)=LOWER(?)",
                (book, book),
            ) as cur:
                row = await cur.fetchone()
        if not row:
            raise ValueError(f"Book not found: {book!r}")
        book_num = row["num"]

    orig_translation = "wlc" if book_num <= 39 else "sblgnt"
    language = "hebrew" if book_num <= 39 else "greek"

    async with get_db() as db:
        book_name_row = await (await db.execute(
            "SELECT name FROM books WHERE num=?", (book_num,)
        )).fetchone()
        book_name = book_name_row["name"] if book_name_row else book

        words = await _interlinear_for_verse(db, book_num, chapter, verse, orig_translation)

    if not words:
        raise ValueError(
            f"No interlinear data for {book} {chapter}:{verse}. "
            f"Ensure {orig_translation} import has been run."
        )

    return InterlinearVerse(
        book_name=book_name,
        chapter=chapter,
        verse=verse,
        language=language,
        words=words,
    )


@mcp.tool(name="xref")
async def cross_reference(book: str, chapter: int, verse: int) -> CrossReference:
    """
    Return all Strong's numbers in a verse with their full lexicon entries.
    Useful for seeing every original-language word behind an English verse.
    """
    book_num = resolve_book(book)
    if not book_num:
        raise ValueError(f"Book not found: {book!r}")

    orig_translation = "wlc" if book_num <= 39 else "sblgnt"

    async with get_db() as db:
        book_name_row = await (await db.execute(
            "SELECT name FROM books WHERE num=?", (book_num,)
        )).fetchone()
        book_name = book_name_row["name"] if book_name_row else book

        words = await _interlinear_for_verse(db, book_num, chapter, verse, orig_translation)

    ref = f"{book_name} {chapter}:{verse}"
    return CrossReference(reference=ref, words=words)
