"""Project bibliography lookup and basic citation-key auditing."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

from .catalog import PaperCatalog
from .models import BibliographyEntry, SchemaValidationError, record_from_dict, record_to_dict
from .project import ResearchContext


class BibliographyError(RuntimeError):
    """Raised when bibliography records are invalid or conflicting."""


@dataclass(frozen=True)
class BibliographyAddResult:
    entry: BibliographyEntry
    created: bool


@dataclass(frozen=True)
class CitationAudit:
    used_keys: tuple[str, ...]
    unknown_keys: tuple[str, ...]
    unused_keys: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.unknown_keys


class BibliographyStore:
    """A project-scoped mapping between paper IDs and citation keys."""

    def __init__(self, context: ResearchContext, path: str | Path | None = None) -> None:
        self.context = context
        self.path = Path(path) if path is not None else context.bibliography_path

    def all(self) -> tuple[BibliographyEntry, ...]:
        if not self.path.exists():
            return ()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise BibliographyError(f"Cannot read bibliography: {self.path}") from exc
        entries: list[BibliographyEntry] = []
        keys: set[str] = set()
        paper_ids: set[str] = set()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = record_from_dict(json.loads(line))
            except (json.JSONDecodeError, SchemaValidationError) as exc:
                raise BibliographyError(
                    f"Invalid bibliography entry at {self.path}:{line_number}"
                ) from exc
            if not isinstance(record, BibliographyEntry):
                raise BibliographyError(
                    f"Non-bibliography record at {self.path}:{line_number}"
                )
            if record.citation_key in keys:
                raise BibliographyError(
                    f"Duplicate citation key in {self.path}: {record.citation_key}"
                )
            if record.paper_id in paper_ids:
                raise BibliographyError(
                    f"Paper has multiple bibliography entries: {record.paper_id}"
                )
            keys.add(record.citation_key)
            paper_ids.add(record.paper_id)
            entries.append(record)
        return tuple(entries)

    def get(self, citation_key: str) -> BibliographyEntry | None:
        return next(
            (entry for entry in self.all() if entry.citation_key == citation_key),
            None,
        )

    def for_paper(self, paper_id: str) -> BibliographyEntry | None:
        return next(
            (entry for entry in self.all() if entry.paper_id == paper_id),
            None,
        )

    def citation_key_for(self, paper_id: str) -> str:
        entry = self.for_paper(paper_id)
        if entry is None:
            raise BibliographyError(f"Paper has no citation key: {paper_id}")
        return entry.citation_key

    def add(self, entry: BibliographyEntry) -> BibliographyAddResult:
        if PaperCatalog(self.context.catalog_path).get(entry.paper_id) is None:
            raise BibliographyError(f"Unknown paper ID: {entry.paper_id}")
        entries = list(self.all())
        by_key = next(
            (item for item in entries if item.citation_key == entry.citation_key),
            None,
        )
        by_paper = next(
            (item for item in entries if item.paper_id == entry.paper_id),
            None,
        )
        existing = by_key or by_paper
        if existing is not None:
            if existing != entry:
                raise BibliographyError(
                    "Citation key or paper already has different bibliography metadata"
                )
            return BibliographyAddResult(entry=existing, created=False)
        entries.append(entry)
        _write_entries(self.path, entries)
        return BibliographyAddResult(entry=entry, created=True)

    def audit_text(self, text: str) -> CitationAudit:
        if not isinstance(text, str):
            raise BibliographyError("Citation audit input must be text")
        groups = re.findall(r"\\cite[a-zA-Z]*\{([^}]+)\}", text)
        used_keys = tuple(
            dict.fromkeys(
                key.strip()
                for group in groups
                for key in group.split(",")
                if key.strip()
            )
        )
        known_keys = {entry.citation_key for entry in self.all()}
        unknown = tuple(key for key in used_keys if key not in known_keys)
        unused = tuple(sorted(known_keys - set(used_keys)))
        return CitationAudit(
            used_keys=used_keys,
            unknown_keys=unknown,
            unused_keys=unused,
        )


def _write_entries(path: Path, entries: list[BibliographyEntry]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "\n".join(
        json.dumps(record_to_dict(entry), ensure_ascii=False, sort_keys=True)
        for entry in entries
    ) + ("\n" if entries else "")
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
        raise BibliographyError(f"Cannot write bibliography: {path}") from exc

