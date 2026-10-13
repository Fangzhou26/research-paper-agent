"""Project-aware paper listing, page reading, and lexical retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

from .catalog import PaperCatalog
from .extract import ExtractionError, load_extracted_pages
from .models import PageText, PaperRecord
from .project import ResearchContext


class RetrievalError(ValueError):
    """Raised when a paper or query cannot be resolved."""


@dataclass(frozen=True)
class PageMatch:
    """A ranked lexical match within one extracted paper page."""

    paper_id: str
    title: str
    page: int
    score: int
    snippet: str


def list_papers(context: ResearchContext) -> tuple[PaperRecord, ...]:
    """List the active project's cataloged papers."""

    return PaperCatalog(context.catalog_path).all()


def read_paper_pages(
    context: ResearchContext,
    paper_id: str,
    pages: Iterable[int],
) -> tuple[PageText, ...]:
    """Read selected one-indexed pages in the caller's requested order."""

    record = _paper(context, paper_id)
    cached_pages = _pages(context, record)
    page_map = {page.page: page for page in cached_pages}
    requested = tuple(pages)
    if not requested:
        raise RetrievalError("At least one page must be requested")
    if any(not isinstance(page, int) or isinstance(page, bool) or page < 1 for page in requested):
        raise RetrievalError("Page numbers must be positive integers")
    missing = [page for page in requested if page not in page_map]
    if missing:
        raise RetrievalError(
            f"Pages outside paper range for {paper_id}: {', '.join(map(str, missing))}"
        )
    return tuple(page_map[page] for page in requested)


def search_paper_pages(
    context: ResearchContext,
    query: str,
    *,
    paper_ids: Iterable[str] | None = None,
    limit: int = 20,
) -> tuple[PageMatch, ...]:
    """Rank extracted pages using transparent phrase and token matching."""

    if not isinstance(query, str) or not query.strip():
        raise RetrievalError("Search query must be a non-empty string")
    if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
        raise RetrievalError("Search limit must be a positive integer")
    normalized_query = " ".join(query.casefold().split())
    terms = tuple(dict.fromkeys(re.findall(r"\w+", normalized_query)))
    selected_ids = set(paper_ids) if paper_ids is not None else None
    records = list_papers(context)
    known_ids = {record.paper_id for record in records}
    if selected_ids is not None:
        unknown = selected_ids - known_ids
        if unknown:
            raise RetrievalError(f"Unknown paper IDs: {', '.join(sorted(unknown))}")
        records = [record for record in records if record.paper_id in selected_ids]

    matches: list[PageMatch] = []
    for record in records:
        for page in _pages(context, record):
            normalized_text = " ".join(page.text.casefold().split())
            phrase_hits = normalized_text.count(normalized_query)
            term_hits = sum(normalized_text.count(term) for term in terms)
            score = phrase_hits * 10 + term_hits
            if score:
                matches.append(
                    PageMatch(
                        paper_id=record.paper_id,
                        title=record.title,
                        page=page.page,
                        score=score,
                        snippet=_snippet(page.text, normalized_query, terms),
                    )
                )
    matches.sort(key=lambda match: (-match.score, match.title.casefold(), match.page))
    return tuple(matches[:limit])


def _paper(context: ResearchContext, paper_id: str) -> PaperRecord:
    record = PaperCatalog(context.catalog_path).get(paper_id)
    if record is None:
        raise RetrievalError(f"Paper is not present in project catalog: {paper_id}")
    return record


def _pages(context: ResearchContext, paper: PaperRecord) -> tuple[PageText, ...]:
    safe_id = paper.paper_id.replace(":", "_")
    cache_path = context.extraction_directory / f"{safe_id}.pages.jsonl"
    try:
        return load_extracted_pages(cache_path, expected_paper_id=paper.paper_id)
    except ExtractionError as exc:
        raise RetrievalError(
            f"Extracted pages are unavailable for {paper.paper_id}"
        ) from exc


def _snippet(text: str, query: str, terms: tuple[str, ...], width: int = 320) -> str:
    compact = " ".join(text.split())
    normalized = compact.casefold()
    position = normalized.find(query)
    if position < 0:
        positions = [normalized.find(term) for term in terms]
        position = min((value for value in positions if value >= 0), default=0)
    start = max(0, position - width // 3)
    end = min(len(compact), start + width)
    prefix = "..." if start else ""
    suffix = "..." if end < len(compact) else ""
    return f"{prefix}{compact[start:end].strip()}{suffix}"

