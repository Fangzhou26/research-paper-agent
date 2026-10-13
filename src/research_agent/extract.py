"""Page-addressable PDF extraction with validated JSONL caching."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Protocol

from .catalog import PaperCatalog
from .models import PageText, PaperRecord, SchemaValidationError, record_from_dict, record_to_dict


class ExtractionError(RuntimeError):
    """Raised when a paper cannot be extracted or its cache is invalid."""


class PdfExtractionBackend(Protocol):
    """Backend contract for converting a PDF into one string per page."""

    name: str

    def extract_pages(self, source_path: Path) -> tuple[str, ...]:
        """Return page text in document order."""


class PypdfBackend:
    """Native PDF text extraction using pypdf."""

    name = "pypdf"

    def extract_pages(self, source_path: Path) -> tuple[str, ...]:
        try:
            from pypdf import PdfReader
        except ImportError as exc:
            raise ExtractionError(
                "pypdf is required for native PDF extraction"
            ) from exc
        try:
            reader = PdfReader(source_path)
            if reader.is_encrypted:
                try:
                    unlocked = reader.decrypt("")
                except Exception as exc:  # pypdf exposes backend-specific errors
                    raise ExtractionError(f"Cannot decrypt PDF: {source_path}") from exc
                if not unlocked:
                    raise ExtractionError(f"PDF requires a password: {source_path}")
            return tuple(page.extract_text() or "" for page in reader.pages)
        except ExtractionError:
            raise
        except Exception as exc:  # normalize pypdf's parser exceptions
            raise ExtractionError(f"Cannot extract PDF: {source_path}") from exc


@dataclass(frozen=True)
class ExtractionResult:
    """Location and quality summary for one extracted paper."""

    paper_id: str
    output_path: Path
    page_count: int
    empty_pages: tuple[int, ...]
    extraction_method: str
    created: bool


def extract_catalog_paper(
    catalog: PaperCatalog,
    paper_id: str,
    output_directory: str | Path,
    *,
    path_base: str | Path | None = None,
    force: bool = False,
    backend: PdfExtractionBackend | None = None,
) -> ExtractionResult:
    """Find a paper in a catalog and extract its pages."""

    record = catalog.get(paper_id)
    if record is None:
        raise ExtractionError(f"Paper is not present in catalog: {paper_id}")
    return extract_paper(
        record,
        output_directory,
        path_base=path_base,
        force=force,
        backend=backend,
    )


def extract_paper(
    paper: PaperRecord,
    output_directory: str | Path,
    *,
    path_base: str | Path | None = None,
    force: bool = False,
    backend: PdfExtractionBackend | None = None,
) -> ExtractionResult:
    """Extract one cataloged paper or return its validated cache."""

    output_path = Path(output_directory) / f"{_safe_id(paper.paper_id)}.pages.jsonl"
    if output_path.exists() and not force:
        pages = load_extracted_pages(output_path, expected_paper_id=paper.paper_id)
        return _result(output_path, pages, created=False)

    source_path = _resolve_source_path(paper, path_base)
    extraction_backend = backend or PypdfBackend()
    try:
        extracted_text = extraction_backend.extract_pages(source_path)
    except ExtractionError:
        raise
    except Exception as exc:
        raise ExtractionError(
            f"Extraction backend '{extraction_backend.name}' failed: {source_path}"
        ) from exc
    if not extracted_text:
        raise ExtractionError(f"PDF contains no pages: {source_path}")
    pages = tuple(
        PageText(
            paper_id=paper.paper_id,
            page=index,
            text=text,
            extraction_method=extraction_backend.name,
            quality=_page_quality(text),
        )
        for index, text in enumerate(extracted_text, start=1)
    )
    _write_pages(output_path, pages)
    return _result(output_path, pages, created=True)


def load_extracted_pages(
    path: str | Path,
    *,
    expected_paper_id: str | None = None,
) -> tuple[PageText, ...]:
    """Load and validate page records from an extraction cache."""

    cache_path = Path(path)
    try:
        lines = cache_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ExtractionError(f"Cannot read extraction cache: {cache_path}") from exc
    pages: list[PageText] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = record_from_dict(json.loads(line))
        except (json.JSONDecodeError, SchemaValidationError) as exc:
            raise ExtractionError(
                f"Invalid extraction record at {cache_path}:{line_number}"
            ) from exc
        if not isinstance(record, PageText):
            raise ExtractionError(
                f"Non-page record at {cache_path}:{line_number}: "
                f"{type(record).__name__}"
            )
        pages.append(record)
    if not pages:
        raise ExtractionError(f"Extraction cache contains no pages: {cache_path}")
    paper_ids = {page.paper_id for page in pages}
    if len(paper_ids) != 1:
        raise ExtractionError(f"Extraction cache mixes paper IDs: {cache_path}")
    if expected_paper_id is not None and paper_ids != {expected_paper_id}:
        raise ExtractionError(
            f"Extraction cache does not match paper ID {expected_paper_id}: {cache_path}"
        )
    expected_pages = list(range(1, len(pages) + 1))
    actual_pages = [page.page for page in pages]
    if actual_pages != expected_pages:
        raise ExtractionError(f"Extraction cache has non-contiguous pages: {cache_path}")
    return tuple(pages)


def _resolve_source_path(paper: PaperRecord, path_base: str | Path | None) -> Path:
    if paper.file_path is None:
        raise ExtractionError(f"Paper has no local PDF path: {paper.paper_id}")
    if paper.file_path.is_absolute():
        source_path = paper.file_path
    elif path_base is not None:
        source_path = Path(path_base).expanduser().resolve() / paper.file_path
    else:
        raise ExtractionError(
            f"Relative PDF path requires path_base: {paper.file_path}"
        )
    if not source_path.is_file():
        raise ExtractionError(f"Paper PDF does not exist: {source_path}")
    return source_path


def _write_pages(path: Path, pages: tuple[PageText, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(record_to_dict(page), ensure_ascii=False, sort_keys=True)
        for page in pages
    ) + "\n"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise ExtractionError(f"Cannot write extraction cache: {path}") from exc


def _page_quality(text: str) -> str:
    character_count = len(text.strip())
    if character_count == 0:
        return "empty"
    if character_count < 100:
        return "low"
    return "good"


def _safe_id(paper_id: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in paper_id
    )


def _result(
    output_path: Path,
    pages: tuple[PageText, ...],
    *,
    created: bool,
) -> ExtractionResult:
    return ExtractionResult(
        paper_id=pages[0].paper_id,
        output_path=output_path,
        page_count=len(pages),
        empty_pages=tuple(page.page for page in pages if page.quality == "empty"),
        extraction_method=pages[0].extraction_method,
        created=created,
    )

