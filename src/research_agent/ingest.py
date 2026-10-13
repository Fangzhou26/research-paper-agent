"""Deterministic ingestion of local PDF papers into a paper catalog."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

from .catalog import PaperCatalog
from .models import PaperRecord


class IngestionError(ValueError):
    """Raised when a source cannot be ingested as a paper."""


@dataclass(frozen=True)
class IngestionResult:
    """A cataloged paper and whether ingestion created a new record."""

    record: PaperRecord
    created: bool


def ingest_paper(
    source: str | Path,
    catalog: PaperCatalog,
    *,
    title: str | None = None,
    authors: tuple[str, ...] = (),
    year: int | None = None,
    venue: str | None = None,
    doi: str | None = None,
    arxiv_id: str | None = None,
    url: str | None = None,
    source_type: str = "uploaded",
    tags: tuple[str, ...] = (),
    path_base: str | Path | None = None,
) -> IngestionResult:
    """Validate and register a local PDF using its SHA-256 content identity.

    The source file is registered in place. Copying files into a managed
    library is deliberately left to a later storage policy.
    """

    source_path = Path(source).expanduser().resolve()
    _validate_pdf(source_path)
    digest = _sha256(source_path)
    paper_id = f"sha256:{digest}"
    existing = catalog.get(paper_id)
    if existing is not None:
        return IngestionResult(record=existing, created=False)

    resolved_title = title.strip() if isinstance(title, str) else source_path.stem
    if not resolved_title:
        raise IngestionError("Paper title cannot be empty")
    stored_path = _stored_path(source_path, path_base)
    record = PaperRecord(
        paper_id=paper_id,
        title=resolved_title,
        authors=authors,
        year=year,
        venue=venue,
        doi=doi,
        arxiv_id=arxiv_id,
        url=url,
        source_type=source_type,
        source_uri=str(stored_path),
        file_path=stored_path,
        verification_status="unverified",
        tags=tags,
    )
    result = catalog.add(record)
    return IngestionResult(record=result.record, created=result.created)


def _stored_path(source_path: Path, path_base: str | Path | None) -> Path:
    if path_base is None:
        return source_path
    base = Path(path_base).expanduser().resolve()
    try:
        return source_path.relative_to(base)
    except ValueError as exc:
        raise IngestionError(
            f"Paper source is outside the configured path base: {source_path}"
        ) from exc


def _validate_pdf(path: Path) -> None:
    if not path.is_file():
        raise IngestionError(f"Paper source does not exist: {path}")
    try:
        with path.open("rb") as source_file:
            header = source_file.read(1024)
    except OSError as exc:
        raise IngestionError(f"Cannot read paper source: {path}") from exc
    if b"%PDF-" not in header:
        raise IngestionError(f"Paper source is not a PDF: {path}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise IngestionError(f"Cannot hash paper source: {path}") from exc
    return digest.hexdigest()
