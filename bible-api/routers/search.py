"""Full-text search tool using SQLite FTS5."""

import re

from database import get_db
from mcp_server import mcp
from models import SearchResult, SearchResultList

MAX_RESULTS = 50
_HIGHLIGHT_RE = re.compile(r"\x02(.*?)\x03", re.DOTALL)


def _make_snippet(text: str, query_terms: list[str], window: int = 40) -> str:
    """Produce a short snippet with query terms wrapped in [brackets]."""
    lower = text.lower()
    best_start = 0
    for term in query_terms:
        idx = lower.find(term.lower())
        if idx >= 0:
            best_start = max(0, idx - window // 2)
            break
    snippet = text[best_start: best_start + window * 2]
    if best_start > 0:
        snippet = "…" + snippet
    if best_start + window * 2 < len(text):
        snippet += "…"
    for term in query_terms:
        snippet = re.sub(
            re.escape(term), f"[{term}]", snippet, flags=re.IGNORECASE
        )
    return snippet


@mcp.tool(name="search")
async def search_bible(
    q: str,
    translation: str | None = None,
    testament: str | None = None,
    limit: int = 10,
) -> SearchResultList:
    """Full-text search the Bible (FTS5 syntax supported). testament is 'OT' or 'NT'; limit caps at 50."""
    if not q.strip() or len(q.strip()) < 2:
        raise ValueError("Query must be at least 2 characters.")
    limit = min(limit, MAX_RESULTS)

    # FTS5 query — escape if it looks like a plain phrase
    fts_query = q if any(c in q for c in ('"', '*', 'AND', 'OR', 'NOT')) else f'"{q}"'

    params: list = [fts_query]
    extra_where = ""

    if translation:
        extra_where += " AND v.translation=?"
        params.append(translation.lower())

    if testament and testament.upper() in ("OT", "NT"):
        extra_where += " AND b.testament=?"
        params.append(testament.upper())

    params.append(limit)

    sql = f"""
        SELECT v.translation, v.book_num, b.name AS book_name,
               v.chapter, v.verse, v.text,
               rank
        FROM verses_fts
        JOIN verses v ON verses_fts.rowid = v.id
        JOIN books  b ON b.num = v.book_num
        WHERE verses_fts MATCH ?
        {extra_where}
        ORDER BY rank
        LIMIT ?
    """

    query_terms = re.findall(r"\w+", q)

    async with get_db() as db:
        try:
            async with db.execute(sql, params) as cur:
                rows = await cur.fetchall()
        except Exception as exc:
            # FTS5 syntax error — retry with quoted literal
            if "fts5" in str(exc).lower() or "syntax" in str(exc).lower():
                params[0] = f'"{q}"'
                async with db.execute(sql, params) as cur:
                    rows = await cur.fetchall()
            else:
                raise ValueError(f"Search error: {exc}")

    results = []
    for row in rows:
        trans_upper = row["translation"].upper()
        ref = f"{row['book_name']} {row['chapter']}:{row['verse']} ({trans_upper})"
        results.append(SearchResult(
            reference=ref,
            translation=row["translation"],
            book_name=row["book_name"],
            chapter=row["chapter"],
            verse=row["verse"],
            text=row["text"],
            snippet=_make_snippet(row["text"], query_terms),
        ))
    return SearchResultList(results=results)
