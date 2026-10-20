"""SQLite connection management using aiosqlite."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite

DB_PATH = Path(os.getenv("DB_PATH", "/data/bibles.db"))

# Book name / abbreviation → book_num lookup (populated at startup)
_book_lookup: dict[str, int] = {}


async def _load_book_lookup(db: aiosqlite.Connection) -> None:
    global _book_lookup
    async with db.execute("SELECT num, name, abbrev FROM books") as cur:
        async for row in cur:
            num, name, abbrev = row
            _book_lookup[name.lower()]   = num
            _book_lookup[abbrev.lower()] = num
            # Common aliases
            if name == "Psalms":
                _book_lookup["psalm"]     = num
                _book_lookup["ps"]        = num
            if name == "Song of Solomon":
                _book_lookup["song of songs"]  = num
                _book_lookup["canticles"]       = num
                _book_lookup["sos"]             = num
                _book_lookup["cant"]            = num
            if name == "Revelation":
                _book_lookup["rev"]       = num
                _book_lookup["apocalypse"]= num
            if name == "Obadiah":
                _book_lookup["ob"]        = num
            if name == "Philemon":
                _book_lookup["phlm"]      = num


def resolve_book(name_or_abbrev: str) -> int | None:
    """Return book_num for a name or abbreviation, or None if not found."""
    return _book_lookup.get(name_or_abbrev.strip().lower())


@asynccontextmanager
async def get_db() -> AsyncIterator[aiosqlite.Connection]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.execute("PRAGMA foreign_keys=ON")
        if not _book_lookup:
            await _load_book_lookup(db)
        yield db


async def startup() -> None:
    """Called at application startup to warm the book lookup cache."""
    if not DB_PATH.exists():
        raise RuntimeError(
            f"Bible database not found at {DB_PATH}. "
            "Provide a SQLite database at DB_PATH before starting."
        )
    async with get_db():
        pass  # just warms _book_lookup
