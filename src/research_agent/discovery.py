"""Seed-grounded academic discovery and corpus expansion."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
from threading import Lock
import time
from typing import Any, Callable, Mapping, Protocol, Sequence
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from xml.etree import ElementTree

from .bibliography import BibliographyStore
from .catalog import PaperCatalog
from .llm import LLMProvider, LLMProviderError
from .models import (
    BibliographyEntry,
    InnovationProfile,
    PageText,
    PaperRecord,
    SearchCandidate,
    record_to_dict,
)
from .project import ResearchContext


class DiscoveryError(RuntimeError):
    """Raised when academic discovery cannot proceed."""


DiscoveryEventSink = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class DiscoveryQuery:
    query: str
    rationale: str


@dataclass(frozen=True)
class DiscoveryResult:
    profile: InnovationProfile
    planner_mode: str
    queries: tuple[DiscoveryQuery, ...]
    candidates: tuple[SearchCandidate, ...]
    accepted_paper_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    report_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": record_to_dict(self.profile),
            "planner_mode": self.planner_mode,
            "queries": [
                {"query": item.query, "rationale": item.rationale}
                for item in self.queries
            ],
            "candidates": [record_to_dict(item) for item in self.candidates],
            "accepted_paper_ids": list(self.accepted_paper_ids),
            "warnings": list(self.warnings),
            "report_path": str(self.report_path),
        }


@dataclass(frozen=True)
class AcademicWork:
    external_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    url: str | None = None
    abstract: str | None = None
    cited_by_count: int = 0
    relevance_score: float = 0.0
    topics: tuple[str, ...] = ()
    related_ids: tuple[str, ...] = ()
    referenced_ids: tuple[str, ...] = ()


class AcademicSearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 10) -> tuple[AcademicWork, ...]: ...

    def neighbors(
        self,
        work: AcademicWork,
        *,
        limit: int = 8,
    ) -> tuple[tuple[AcademicWork, str], ...]: ...


class DiscoveryQueryPlanner:
    """Use the configured model for query decomposition with a safe fallback."""

    def __init__(self, provider: LLMProvider, *, max_queries: int = 5) -> None:
        self.provider = provider
        self.max_queries = max_queries

    def plan(
        self,
        profile: InnovationProfile,
        seed_titles: Sequence[str],
    ) -> tuple[tuple[DiscoveryQuery, ...], str]:
        messages = [
            {
                "role": "system",
                "content": (
                    "Decompose a seed-grounded literature search into 3 to 5 concise "
                    "academic search queries. Cover the core topic, method families, "
                    "and the problems addressed by the proposed innovations. Do not "
                    "invent paper titles. Return JSON only as {\"queries\": "
                    "[{\"query\": str, \"rationale\": str}]} ."
                ),
            },
            {
                "role": "user",
                "content": json.dumps({
                    "topic": profile.topic,
                    "innovation_points": list(profile.innovation_points),
                    "seed_papers": list(seed_titles),
                    "constraints": list(profile.constraints),
                }, ensure_ascii=False),
            },
        ]
        try:
            value = self.provider.generate_json(messages)
            queries = _parse_queries(value, self.max_queries)
            return queries, "llm"
        except (LLMProviderError, DiscoveryError):
            return _fallback_queries(profile, seed_titles, self.max_queries), "fallback"


class OpenAlexProvider:
    """Small adapter over the official OpenAlex works API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.openalex.org",
        request_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
        timeout: float = 20,
    ) -> None:
        self.api_key = api_key or os.environ.get("OPENALEX_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._request_json = request_json or self._default_request

    def search(self, query: str, *, limit: int = 10) -> tuple[AcademicWork, ...]:
        if not query.strip():
            raise DiscoveryError("Discovery query must not be empty")
        payload = self._request("/works", {
            "search": query.strip(),
            "per_page": str(max(1, min(limit, 50))),
            "select": _OPENALEX_SELECT,
        })
        return tuple(_work(item) for item in _results(payload))

    def neighbors(
        self,
        work: AcademicWork,
        *,
        limit: int = 8,
    ) -> tuple[tuple[AcademicWork, str], ...]:
        found: list[tuple[AcademicWork, str]] = []
        related = self._many(work.related_ids[:limit])
        found.extend((item, "related_to_seed") for item in related)
        references = self._many(work.referenced_ids[:limit])
        found.extend((item, "referenced_by_seed") for item in references)
        openalex_id = work.external_id.rsplit("/", 1)[-1]
        payload = self._request("/works", {
            "filter": f"cites:{openalex_id}",
            "sort": "cited_by_count:desc",
            "per_page": str(max(1, min(limit, 50))),
            "select": _OPENALEX_SELECT,
        })
        found.extend((_work(item), "cites_seed") for item in _results(payload))
        return tuple(found)

    def _many(self, identifiers: Sequence[str]) -> tuple[AcademicWork, ...]:
        ids = [item.rsplit("/", 1)[-1] for item in identifiers if item]
        if not ids:
            return ()
        payload = self._request("/works", {
            "filter": "openalex_id:" + "|".join(ids),
            "per_page": str(min(len(ids), 50)),
            "select": _OPENALEX_SELECT,
        })
        return tuple(_work(item) for item in _results(payload))

    def _request(self, path: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        values = dict(params)
        if self.api_key:
            values["api_key"] = self.api_key
        try:
            return self._request_json(path, values)
        except DiscoveryError:
            raise
        except Exception as exc:
            raise DiscoveryError(f"OpenAlex request failed: {exc}") from exc

    def _default_request(self, path: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        request = Request(url, headers={"User-Agent": "research-agent/0.1"})
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"OpenAlex request failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DiscoveryError("OpenAlex returned a non-object response")
        return value


class SemanticScholarProvider:
    """Adapter for the official Semantic Scholar Academic Graph API."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = "https://api.semanticscholar.org/graph/v1",
        request_json: Callable[[str, Mapping[str, str]], Mapping[str, Any]] | None = None,
        timeout: float = 20,
    ) -> None:
        self.api_key = api_key or os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._request_json = request_json or self._default_request

    def search(self, query: str, *, limit: int = 10) -> tuple[AcademicWork, ...]:
        payload = self._request("/paper/search", {
            "query": query.strip(),
            "limit": str(max(1, min(limit, 100))),
            "fields": _S2_FIELDS,
        })
        data = payload.get("data")
        if not isinstance(data, list):
            raise DiscoveryError("Semantic Scholar search response has no data array")
        return tuple(_s2_work(item) for item in data if isinstance(item, Mapping))

    def neighbors(
        self,
        work: AcademicWork,
        *,
        limit: int = 8,
    ) -> tuple[tuple[AcademicWork, str], ...]:
        paper_id = work.external_id.removeprefix("S2:")
        found: list[tuple[AcademicWork, str]] = []
        for relation, nested, source in (
            ("references", "citedPaper", "referenced_by_seed"),
            ("citations", "citingPaper", "cites_seed"),
        ):
            payload = self._request(f"/paper/{quote(paper_id, safe='')}/{relation}", {
                "limit": str(max(1, min(limit, 100))),
                "fields": _S2_FIELDS,
            })
            data = payload.get("data")
            if not isinstance(data, list):
                continue
            for item in data:
                paper = item.get(nested) if isinstance(item, Mapping) else None
                if isinstance(paper, Mapping) and paper.get("paperId"):
                    found.append((_s2_work(paper), source))
        return tuple(found)

    def _request(self, path: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        try:
            return self._request_json(path, params)
        except DiscoveryError:
            raise
        except Exception as exc:
            raise DiscoveryError(f"Semantic Scholar request failed: {exc}") from exc

    def _default_request(self, path: str, params: Mapping[str, str]) -> Mapping[str, Any]:
        url = f"{self.base_url}{path}?{urlencode(params)}"
        headers = {"User-Agent": "research-agent/0.1"}
        if self.api_key:
            headers["x-api-key"] = self.api_key
        request = Request(url, headers=headers)
        try:
            with urlopen(request, timeout=self.timeout) as response:
                value = json.load(response)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise DiscoveryError(f"Semantic Scholar request failed: {exc}") from exc
        if not isinstance(value, Mapping):
            raise DiscoveryError("Semantic Scholar returned a non-object response")
        return value


class FallbackAcademicProvider:
    """Try multiple academic indexes without hiding total source failure."""

    def __init__(self, providers: Sequence[AcademicSearchProvider]) -> None:
        if not providers:
            raise ValueError("At least one academic provider is required")
        self.providers = tuple(providers)

    def search(self, query: str, *, limit: int = 10) -> tuple[AcademicWork, ...]:
        works: dict[str, AcademicWork] = {}
        errors: list[str] = []
        for provider in self.providers:
            try:
                results = provider.search(query, limit=limit)
            except DiscoveryError as exc:
                errors.append(str(exc))
                continue
            for work in results:
                key = _doi(work.doi) if work.doi else _normalize_title(work.title)
                works.setdefault(key or work.external_id, work)
        if not works and errors:
            raise DiscoveryError("; ".join(errors))
        return tuple(works.values())[:limit]

    def neighbors(
        self,
        work: AcademicWork,
        *,
        limit: int = 8,
    ) -> tuple[tuple[AcademicWork, str], ...]:
        ordered = sorted(
            self.providers,
            key=lambda provider: not (
                work.external_id.startswith("S2:")
                and isinstance(provider, SemanticScholarProvider)
                or work.external_id.startswith("https://openalex.org/")
                and isinstance(provider, OpenAlexProvider)
                or work.external_id.startswith("ARXIV:")
                and isinstance(provider, ArxivProvider)
            ),
        )
        errors: list[str] = []
        for provider in ordered:
            try:
                return provider.neighbors(work, limit=limit)
            except DiscoveryError as exc:
                errors.append(str(exc))
        raise DiscoveryError("; ".join(errors))


class ArxivProvider:
    """No-key abstract-search fallback for arXiv-covered research domains."""

    def __init__(
        self,
        *,
        base_url: str = "https://export.arxiv.org/api/query",
        request_text: Callable[[Mapping[str, str]], str] | None = None,
        timeout: float = 20,
        min_interval: float = 3.0,
    ) -> None:
        self.base_url = base_url
        self.timeout = timeout
        self.min_interval = max(0, min_interval)
        self._request_lock = Lock()
        self._last_request_at = 0.0
        self._request_text = request_text or self._default_request

    def search(self, query: str, *, limit: int = 10) -> tuple[AcademicWork, ...]:
        tokens = re.findall(r"[A-Za-z0-9]+", query)
        if not tokens:
            raise DiscoveryError("arXiv query must contain searchable terms")
        compact = " AND ".join(f"all:{token}" for token in tokens[:12])
        return self._query({
            "search_query": compact,
            "start": "0",
            "max_results": str(max(1, min(limit, 50))),
            "sortBy": "relevance",
            "sortOrder": "descending",
        })

    def neighbors(
        self,
        work: AcademicWork,
        *,
        limit: int = 8,
    ) -> tuple[tuple[AcademicWork, str], ...]:
        related = self.search(work.title, limit=limit + 1)
        return tuple(
            (item, "title_related_to_seed")
            for item in related
            if item.external_id != work.external_id
        )[:limit]

    def _query(self, params: Mapping[str, str]) -> tuple[AcademicWork, ...]:
        try:
            with self._request_lock:
                remaining = self.min_interval - (time.monotonic() - self._last_request_at)
                if remaining > 0:
                    time.sleep(remaining)
                xml = self._request_text(params)
                self._last_request_at = time.monotonic()
            root = ElementTree.fromstring(xml)
        except (DiscoveryError, ElementTree.ParseError) as exc:
            raise DiscoveryError(f"arXiv request failed: {exc}") from exc
        namespace = {"atom": "http://www.w3.org/2005/Atom"}
        return tuple(_arxiv_work(entry, namespace) for entry in root.findall("atom:entry", namespace))

    def _default_request(self, params: Mapping[str, str]) -> str:
        request = Request(
            f"{self.base_url}?{urlencode(params)}",
            headers={
                "User-Agent": "research-agent/0.1",
                "Accept": "application/atom+xml",
            },
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, UnicodeDecodeError) as exc:
            raise DiscoveryError(f"arXiv request failed: {exc}") from exc


class SeedDiscoveryPipeline:
    """Expand seed papers, screen candidates, and register an evidence corpus."""

    def __init__(
        self,
        context: ResearchContext,
        academic_provider: AcademicSearchProvider,
        query_planner: DiscoveryQueryPlanner,
        *,
        on_event: DiscoveryEventSink | None = None,
    ) -> None:
        self.context = context
        self.academic_provider = academic_provider
        self.query_planner = query_planner
        self.on_event = on_event

    def run(
        self,
        profile: InnovationProfile,
        *,
        max_papers: int = 12,
        per_query: int = 8,
    ) -> DiscoveryResult:
        if profile.project_id != self.context.project_id:
            raise DiscoveryError("Innovation profile belongs to another project")
        if max_papers < len(profile.seed_paper_ids):
            raise DiscoveryError("max_papers must leave room for every seed paper")
        catalog = PaperCatalog(self.context.catalog_path)
        seed_papers = []
        for paper_id in profile.seed_paper_ids:
            paper = catalog.get(paper_id)
            if paper is None or paper.file_path is None:
                raise DiscoveryError(f"Uploaded seed paper is not in the local corpus: {paper_id}")
            seed_papers.append(paper)
        self._emit("discovery_started", {
            "topic": profile.topic,
            "seed_paper_ids": list(profile.seed_paper_ids),
        })

        works: dict[str, tuple[AcademicWork, set[str]]] = {}
        warnings: list[str] = []
        seed_titles = [paper.title for paper in seed_papers]
        for paper in seed_papers:
            self._emit("seed_loaded", {
                "paper_id": paper.paper_id,
                "title": paper.title,
                "source": "uploaded_pdf",
            })

        queries, planner_mode = self.query_planner.plan(profile, seed_titles)
        normalized_seed_titles = {_normalize_title(title) for title in seed_titles}
        self._emit("queries_planned", {
            "planner_mode": planner_mode,
            "queries": [item.query for item in queries],
        })
        for query in queries:
            try:
                results = self.academic_provider.search(query.query, limit=per_query)
            except DiscoveryError as exc:
                results = ()
                warnings.append(f"Search failed for {query.query}: {exc}")
                self._emit("discovery_source_failed", {
                    "stage": "query_search",
                    "query": query.query,
                    "message": str(exc),
                })
            for work in results:
                _merge_work(works, work, f"query:{query.query}")
                if _normalize_title(work.title) in normalized_seed_titles:
                    self._expand_neighbors(work, works, warnings)
            self._emit("search_completed", {
                "query": query.query,
                "result_count": len(results),
            })

        candidates = _screen(
            profile,
            works,
            max_papers=max_papers - len(profile.seed_paper_ids),
        )
        accepted_ids: list[str] = list(profile.seed_paper_ids)
        for candidate in candidates:
            if candidate.screening_decision != "accepted":
                continue
            work, _ = works[candidate.external_id or candidate.candidate_id]
            paper_id = _register_work(self.context, work)
            if paper_id is not None:
                accepted_ids.append(paper_id)
        report_path = self.context.project_directory / "discovery_report.json"
        result = DiscoveryResult(
            profile=profile,
            planner_mode=planner_mode,
            queries=queries,
            candidates=candidates,
            accepted_paper_ids=tuple(dict.fromkeys(accepted_ids)),
            warnings=tuple(warnings),
            report_path=report_path,
        )
        report_path.write_text(
            json.dumps(result.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        self._emit("screening_completed", {
            "candidate_count": len(candidates),
            "accepted_count": len(result.accepted_paper_ids),
        })
        return result

    def _expand_neighbors(
        self,
        work: AcademicWork,
        works: dict[str, tuple[AcademicWork, set[str]]],
        warnings: list[str],
    ) -> None:
        try:
            neighbors = self.academic_provider.neighbors(work)
        except DiscoveryError as exc:
            warnings.append(f"Citation expansion failed for {work.title}: {exc}")
            self._emit("discovery_source_failed", {
                "stage": "citation_expansion",
                "paper": work.title,
                "message": str(exc),
            })
            return
        for neighbor, source in neighbors:
            _merge_work(works, neighbor, source)

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event_type, data)


_OPENALEX_SELECT = ",".join((
    "id", "display_name", "authorships", "publication_year", "primary_location",
    "doi", "abstract_inverted_index", "cited_by_count", "relevance_score",
    "topics", "related_works", "referenced_works",
))
_S2_FIELDS = ",".join((
    "paperId", "title", "abstract", "url", "year", "authors", "venue",
    "citationCount", "externalIds", "fieldsOfStudy",
))


def _parse_queries(value: Mapping[str, Any], max_queries: int) -> tuple[DiscoveryQuery, ...]:
    raw = value.get("queries")
    if not isinstance(raw, list) or not 1 <= len(raw) <= max_queries:
        raise DiscoveryError("Query planner returned an invalid query count")
    queries: list[DiscoveryQuery] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise DiscoveryError("Query planner returned a non-object query")
        query = item.get("query")
        rationale = item.get("rationale")
        if not isinstance(query, str) or not query.strip():
            raise DiscoveryError("Query planner returned an empty query")
        if not isinstance(rationale, str) or not rationale.strip():
            raise DiscoveryError("Query planner returned an empty rationale")
        queries.append(DiscoveryQuery(query.strip(), rationale.strip()))
    return tuple(queries)


def _fallback_queries(
    profile: InnovationProfile,
    seed_titles: Sequence[str],
    max_queries: int,
) -> tuple[DiscoveryQuery, ...]:
    values = [
        DiscoveryQuery(profile.topic, "Core topic search"),
        *(
            DiscoveryQuery(
                f"{profile.topic} {point}",
                "Connect the topic to a proposed innovation",
            )
            for point in profile.innovation_points
        ),
        *(
            DiscoveryQuery(title, "Recover work adjacent to a seed paper")
            for title in seed_titles
        ),
    ]
    unique: list[DiscoveryQuery] = []
    seen: set[str] = set()
    for item in values:
        key = item.query.casefold()
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return tuple(unique[:max_queries])


def _merge_work(
    works: dict[str, tuple[AcademicWork, set[str]]],
    work: AcademicWork,
    source: str,
) -> None:
    key = work.external_id
    if key in works:
        works[key][1].add(source)
        if work.relevance_score > works[key][0].relevance_score:
            works[key] = (work, works[key][1])
    else:
        works[key] = (work, {source})


def _screen(
    profile: InnovationProfile,
    works: Mapping[str, tuple[AcademicWork, set[str]]],
    *,
    max_papers: int,
) -> tuple[SearchCandidate, ...]:
    target_tokens = _tokens(" ".join((profile.topic, *profile.innovation_points)))
    ranked: list[tuple[float, AcademicWork, set[str]]] = []
    for work, sources in works.values():
        text_tokens = _tokens(" ".join((work.title, work.abstract or "", *work.topics)))
        overlap = len(target_tokens & text_tokens) / max(1, len(target_tokens))
        graph_bonus = 0.2 if any("seed" in source for source in sources) else 0.0
        seed_bonus = 1.0 if "seed" in sources else 0.0
        citation_signal = min(math.log1p(work.cited_by_count) / 20, 0.35)
        provider_signal = min(max(work.relevance_score, 0.0) / 100, 0.35)
        score = seed_bonus + overlap + graph_bonus + citation_signal + provider_signal
        ranked.append((score, work, sources))
    ranked.sort(key=lambda item: (
        0 if "seed" in item[2] else 1,
        -item[0],
        -(item[1].year or 0),
        item[1].title.casefold(),
    ))

    candidates: list[SearchCandidate] = []
    accepted = 0
    for score, work, sources in ranked:
        has_readable_source = bool(work.abstract and work.abstract.strip())
        decision = "accepted" if accepted < max_papers and has_readable_source else "rejected"
        if decision == "accepted":
            accepted += 1
            reason = "Selected by seed/query relevance with readable abstract evidence."
        elif not has_readable_source:
            reason = "Metadata found, but no abstract or local full text is available."
        else:
            reason = "Below the configured corpus-size cutoff."
        candidates.append(SearchCandidate(
            candidate_id=_provider_paper_id(work),
            title=work.title,
            authors=work.authors,
            year=work.year,
            source_url=work.url,
            external_id=work.external_id,
            abstract=work.abstract,
            doi=work.doi,
            venue=work.venue,
            cited_by_count=work.cited_by_count,
            relevance_score=round(score, 4),
            discovery_sources=tuple(sorted(sources)),
            topics=work.topics,
            screening_decision=decision,
            screening_reason=reason,
        ))
    return tuple(candidates)


def _register_work(context: ResearchContext, work: AcademicWork) -> str | None:
    if not work.abstract or not work.abstract.strip():
        return None
    catalog = PaperCatalog(context.catalog_path)
    normalized_title = _normalize_title(work.title)
    existing = next(
        (
            paper for paper in catalog.all()
            if _normalize_title(paper.title) == normalized_title
            or (work.doi and paper.doi and _doi(paper.doi) == _doi(work.doi))
        ),
        None,
    )
    if existing is not None:
        return existing.paper_id
    paper_id = _provider_paper_id(work)
    record = PaperRecord(
        paper_id=paper_id,
        title=work.title,
        authors=work.authors,
        year=work.year,
        venue=work.venue,
        doi=work.doi,
        url=work.url,
        source_type="academic_index",
        source_uri=work.external_id,
        verification_status="academic_index_metadata",
        tags=work.topics[:8],
    )
    catalog.add(record)
    context.extraction_directory.mkdir(parents=True, exist_ok=True)
    extraction_path = context.extraction_directory / f"{paper_id.replace(':', '_')}.pages.jsonl"
    page = PageText(
        paper_id=paper_id,
        page=1,
        text=work.abstract,
        extraction_method="academic_index_abstract",
        quality="metadata_abstract",
    )
    extraction_path.write_text(
        json.dumps(record_to_dict(page), ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    bibliography = BibliographyStore(context)
    key = _citation_key(work, {item.citation_key for item in bibliography.all()})
    fields = [
        ("title", work.title),
        ("author", " and ".join(work.authors) or "Unknown"),
        ("year", str(work.year or "n.d.")),
    ]
    if work.venue:
        fields.append(("journal", work.venue))
    if work.doi:
        fields.append(("doi", work.doi))
    if work.url:
        fields.append(("url", work.url))
    bibliography.add(BibliographyEntry(
        citation_key=key,
        paper_id=paper_id,
        entry_type="article",
        fields=tuple(fields),
        verification_status="academic_index_metadata",
    ))
    return paper_id


def _citation_key(work: AcademicWork, known: set[str]) -> str:
    author = work.authors[0].split()[-1] if work.authors else "unknown"
    author = re.sub(r"[^a-z0-9]", "", author.casefold()) or "unknown"
    title_words = [word for word in re.findall(r"[a-z0-9]+", work.title.casefold()) if len(word) > 3]
    stem = f"{author}{work.year or 'nd'}{title_words[0] if title_words else 'work'}"
    if stem not in known:
        return stem
    suffix = re.sub(r"\D", "", work.external_id)[-4:] or hashlib.sha1(
        work.external_id.encode("utf-8")
    ).hexdigest()[:4]
    return f"{stem}{suffix}"


def _provider_paper_id(work: AcademicWork) -> str:
    if work.external_id.startswith("S2:"):
        return "semanticscholar:" + work.external_id.removeprefix("S2:")
    if work.external_id.startswith("ARXIV:"):
        return "arxiv:" + work.external_id.removeprefix("ARXIV:")
    return "openalex:" + work.external_id.rsplit("/", 1)[-1]


def _work(value: Mapping[str, Any]) -> AcademicWork:
    external_id = value.get("id")
    title = value.get("display_name") or value.get("title")
    if not isinstance(external_id, str) or not isinstance(title, str):
        raise DiscoveryError("OpenAlex work is missing identity or title")
    authors = tuple(
        name
        for item in value.get("authorships") or []
        if isinstance(item, Mapping)
        and isinstance(item.get("author"), Mapping)
        and isinstance((name := item["author"].get("display_name")), str)
        and name.strip()
    )
    location = value.get("primary_location")
    location = location if isinstance(location, Mapping) else {}
    source = location.get("source")
    source = source if isinstance(source, Mapping) else {}
    topics = tuple(
        name
        for item in value.get("topics") or []
        if isinstance(item, Mapping)
        and isinstance((name := item.get("display_name")), str)
        and name.strip()
    )
    doi = _doi(value.get("doi")) if isinstance(value.get("doi"), str) else None
    year = value.get("publication_year")
    cited = value.get("cited_by_count", 0)
    relevance = value.get("relevance_score", 0.0)
    return AcademicWork(
        external_id=external_id,
        title=title.strip(),
        authors=authors,
        year=year if isinstance(year, int) else None,
        venue=source.get("display_name") if isinstance(source.get("display_name"), str) else None,
        doi=doi,
        url=location.get("landing_page_url") if isinstance(location.get("landing_page_url"), str) else external_id,
        abstract=_abstract(value.get("abstract_inverted_index")),
        cited_by_count=cited if isinstance(cited, int) and cited >= 0 else 0,
        relevance_score=float(relevance) if isinstance(relevance, (int, float)) else 0.0,
        topics=topics,
        related_ids=tuple(item for item in value.get("related_works") or [] if isinstance(item, str)),
        referenced_ids=tuple(item for item in value.get("referenced_works") or [] if isinstance(item, str)),
    )


def _s2_work(value: Mapping[str, Any]) -> AcademicWork:
    paper_id = value.get("paperId")
    title = value.get("title")
    if not isinstance(paper_id, str) or not isinstance(title, str):
        raise DiscoveryError("Semantic Scholar paper is missing identity or title")
    authors = tuple(
        name
        for item in value.get("authors") or []
        if isinstance(item, Mapping)
        and isinstance((name := item.get("name")), str)
        and name.strip()
    )
    external_ids = value.get("externalIds")
    external_ids = external_ids if isinstance(external_ids, Mapping) else {}
    doi_value = external_ids.get("DOI")
    year = value.get("year")
    citations = value.get("citationCount", 0)
    match_score = value.get("matchScore", 0.0)
    abstract = value.get("abstract")
    return AcademicWork(
        external_id=f"S2:{paper_id}",
        title=title.strip(),
        authors=authors,
        year=year if isinstance(year, int) else None,
        venue=value.get("venue") if isinstance(value.get("venue"), str) else None,
        doi=_doi(doi_value) if isinstance(doi_value, str) else None,
        url=value.get("url") if isinstance(value.get("url"), str) else None,
        abstract=abstract.strip() if isinstance(abstract, str) and abstract.strip() else None,
        cited_by_count=citations if isinstance(citations, int) and citations >= 0 else 0,
        relevance_score=float(match_score) if isinstance(match_score, (int, float)) else 0.0,
        topics=tuple(
            item for item in value.get("fieldsOfStudy") or []
            if isinstance(item, str) and item.strip()
        ),
    )


def _arxiv_work(entry: ElementTree.Element, namespace: Mapping[str, str]) -> AcademicWork:
    identifier = entry.findtext("atom:id", default="", namespaces=namespace).strip()
    match = re.search(r"/abs/([^v]+)(?:v\d+)?$", identifier)
    arxiv_id = match.group(1) if match else identifier.rsplit("/", 1)[-1]
    title = " ".join(entry.findtext("atom:title", default="", namespaces=namespace).split())
    abstract = " ".join(entry.findtext("atom:summary", default="", namespaces=namespace).split())
    if not arxiv_id or not title:
        raise DiscoveryError("arXiv entry is missing identity or title")
    authors = tuple(
        " ".join(author.findtext("atom:name", default="", namespaces=namespace).split())
        for author in entry.findall("atom:author", namespace)
        if author.findtext("atom:name", default="", namespaces=namespace).strip()
    )
    published = entry.findtext("atom:published", default="", namespaces=namespace)
    year = int(published[:4]) if re.match(r"\d{4}", published) else None
    arxiv_namespace = {"arxiv": "http://arxiv.org/schemas/atom"}
    doi = entry.findtext("arxiv:doi", default="", namespaces=arxiv_namespace).strip() or None
    venue = entry.findtext(
        "arxiv:journal_ref", default="", namespaces=arxiv_namespace
    ).strip() or "arXiv"
    topics = tuple(
        term
        for category in entry.findall("atom:category", namespace)
        if isinstance((term := category.attrib.get("term")), str) and term
    )
    return AcademicWork(
        external_id=f"ARXIV:{arxiv_id}",
        title=title,
        authors=authors,
        year=year,
        venue=venue,
        doi=_doi(doi) if doi else None,
        url=identifier or f"https://arxiv.org/abs/{arxiv_id}",
        abstract=abstract or None,
        topics=topics,
    )


def _abstract(value: Any) -> str | None:
    if not isinstance(value, Mapping):
        return None
    positions: list[tuple[int, str]] = []
    for word, indexes in value.items():
        if not isinstance(word, str) or not isinstance(indexes, list):
            continue
        positions.extend((index, word) for index in indexes if isinstance(index, int))
    if not positions:
        return None
    return " ".join(word for _, word in sorted(positions))


def _results(value: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    results = value.get("results")
    if not isinstance(results, list):
        raise DiscoveryError("OpenAlex list response has no results array")
    return [item for item in results if isinstance(item, Mapping)]


def _doi(value: str) -> str | None:
    text = value.strip().casefold()
    text = text.removeprefix("https://doi.org/").removeprefix("http://doi.org/")
    text = text.removeprefix("doi:")
    return text if text.startswith("10.") and "/" in text else None


def _normalize_title(value: str) -> str:
    return " ".join(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _tokens(value: str) -> set[str]:
    return {
        token for token in re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE)
        if len(token) > 2
    }
