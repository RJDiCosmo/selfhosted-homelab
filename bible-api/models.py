"""Pydantic response models for the Bible API."""

from pydantic import BaseModel


class Translation(BaseModel):
    id: str
    name: str
    language: str
    direction: str


class Book(BaseModel):
    num: int
    name: str
    abbrev: str
    testament: str
    chapters: int


class Verse(BaseModel):
    translation: str
    book_num: int
    book_name: str
    chapter: int
    verse: int
    text: str
    reference: str  # e.g. "John 3:16 (KJV)"


class SearchResult(BaseModel):
    reference: str
    translation: str
    book_name: str
    chapter: int
    verse: int
    text: str
    snippet: str  # text with search terms highlighted (brackets)


class LexiconEntry(BaseModel):
    strongs_num: str
    language: str
    word: str
    transliteration: str | None
    pronunciation: str | None
    definition: str
    kjv_usage: str | None


class InterlinearWord(BaseModel):
    position: int
    word: str          # original language word
    strongs_num: str | None
    transliteration: str | None
    definition: str | None
    morphology: str | None


class InterlinearVerse(BaseModel):
    book_name: str
    chapter: int
    verse: int
    language: str
    words: list[InterlinearWord]


class CrossReference(BaseModel):
    """All Strong's numbers in a verse with their lexicon entries."""
    reference: str
    words: list[InterlinearWord]


# Wrapper models for list-returning tools: FastMCP serializes a bare list[Model]
# return as one MCP content block *per item*, which can blow past a small model's
# ability to track many separate tool-result blocks in one turn. Wrapping the list
# in a single object keeps each tool call to one content block.
class TranslationList(BaseModel):
    translations: list[Translation]


class BookList(BaseModel):
    books: list[Book]


class VerseList(BaseModel):
    verses: list[Verse]


class SearchResultList(BaseModel):
    results: list[SearchResult]


class OccurrenceList(BaseModel):
    occurrences: list[dict]
