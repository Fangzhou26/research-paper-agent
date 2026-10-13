"""Project-scoped stores for structured paper notes and page evidence."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from .catalog import PaperCatalog
from .models import (
    EvidenceRecord,
    PaperNote,
    SchemaValidationError,
    record_from_dict,
    record_to_dict,
)
from .project import ResearchContext
from .retrieval import RetrievalError, read_paper_pages


class NoteStoreError(RuntimeError):
    """Raised when notes or evidence cannot be validated or persisted."""


class NoteConflictError(NoteStoreError):
    """Raised when an existing record would be overwritten implicitly."""


@dataclass(frozen=True)
class StoreWriteResult:
    """Result of persisting a note or evidence record."""

    created: bool
    updated: bool


class EvidenceStore:
    """Appendable project evidence with page-level validation."""

    def __init__(self, context: ResearchContext, path: str | Path | None = None) -> None:
        self.context = context
        self.path = Path(path) if path is not None else context.evidence_path

    def all(self) -> tuple[EvidenceRecord, ...]:
        records = _load_records(self.path, EvidenceRecord)
        evidence_ids: set[str] = set()
        for record in records:
            if record.evidence_id in evidence_ids:
                raise NoteStoreError(
                    f"Duplicate evidence ID in {self.path}: {record.evidence_id}"
                )
            evidence_ids.add(record.evidence_id)
        return records

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        return next(
            (record for record in self.all() if record.evidence_id == evidence_id),
            None,
        )

    def list(self, *, paper_id: str | None = None) -> tuple[EvidenceRecord, ...]:
        records = self.all()
        if paper_id is None:
            return records
        return tuple(record for record in records if record.paper_id == paper_id)

    def add(self, evidence: EvidenceRecord) -> StoreWriteResult:
        self._validate(evidence)
        records = list(self.all())
        existing = next(
            (
                record
                for record in records
                if record.evidence_id == evidence.evidence_id
            ),
            None,
        )
        if existing is not None:
            if existing != evidence:
                raise NoteConflictError(
                    f"Evidence ID already has different content: {evidence.evidence_id}"
                )
            return StoreWriteResult(created=False, updated=False)
        records.append(evidence)
        _write_jsonl(self.path, records)
        return StoreWriteResult(created=True, updated=False)

    def _validate(self, evidence: EvidenceRecord) -> None:
        if evidence.project_id != self.context.project_id:
            raise NoteStoreError(
                f"Evidence belongs to another project: {evidence.project_id}"
            )
        if PaperCatalog(self.context.catalog_path).get(evidence.paper_id) is None:
            raise NoteStoreError(f"Unknown paper ID: {evidence.paper_id}")
        try:
            page = read_paper_pages(
                self.context,
                evidence.paper_id,
                [evidence.page],
            )[0]
        except RetrievalError as exc:
            raise NoteStoreError(str(exc)) from exc
        passage = " ".join(evidence.passage.split())
        page_text = " ".join(page.text.split())
        if passage not in page_text:
            raise NoteStoreError(
                f"Evidence passage is not present on page {evidence.page} "
                f"of {evidence.paper_id}"
            )


class PaperNoteStore:
    """One validated structured note per project paper."""

    def __init__(
        self,
        context: ResearchContext,
        directory: str | Path | None = None,
        evidence_store: EvidenceStore | None = None,
    ) -> None:
        self.context = context
        self.directory = (
            Path(directory) if directory is not None else context.notes_directory
        )
        self.evidence_store = evidence_store or EvidenceStore(context)

    def load(self, paper_id: str) -> PaperNote | None:
        path = self._path(paper_id)
        if not path.exists():
            return None
        records = _load_records(path, PaperNote)
        if len(records) != 1:
            raise NoteStoreError(f"Paper note must contain one record: {path}")
        return records[0]

    def all(self) -> tuple[PaperNote, ...]:
        if not self.directory.exists():
            return ()
        notes: list[PaperNote] = []
        for path in sorted(self.directory.glob("*.json")):
            records = _load_records(path, PaperNote)
            if len(records) != 1:
                raise NoteStoreError(f"Paper note must contain one record: {path}")
            notes.append(records[0])
        return tuple(notes)

    def save(self, note: PaperNote, *, overwrite: bool = False) -> StoreWriteResult:
        self._validate(note)
        existing = self.load(note.paper_id)
        if existing is not None:
            if existing == note:
                return StoreWriteResult(created=False, updated=False)
            if not overwrite:
                raise NoteConflictError(
                    f"Paper note already exists with different content: {note.paper_id}"
                )
        path = self._path(note.paper_id)
        _write_text_atomic(
            path,
            json.dumps(record_to_dict(note), ensure_ascii=False, sort_keys=True) + "\n",
        )
        return StoreWriteResult(created=existing is None, updated=existing is not None)

    def _validate(self, note: PaperNote) -> None:
        if note.project_id != self.context.project_id:
            raise NoteStoreError(f"Note belongs to another project: {note.project_id}")
        if PaperCatalog(self.context.catalog_path).get(note.paper_id) is None:
            raise NoteStoreError(f"Unknown paper ID: {note.paper_id}")
        known_evidence = {
            evidence.evidence_id for evidence in self.evidence_store.all()
        }
        missing = set(note.evidence_ids) - known_evidence
        if missing:
            raise NoteStoreError(
                f"Unknown evidence IDs in note: {', '.join(sorted(missing))}"
            )

    def _path(self, paper_id: str) -> Path:
        return self.directory / f"{_safe_id(paper_id)}.json"


def _load_records(path: Path, expected_type):
    if not path.exists():
        return ()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise NoteStoreError(f"Cannot read research records: {path}") from exc
    records = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = record_from_dict(json.loads(line))
        except (json.JSONDecodeError, SchemaValidationError) as exc:
            raise NoteStoreError(f"Invalid record at {path}:{line_number}") from exc
        if not isinstance(record, expected_type):
            raise NoteStoreError(
                f"Unexpected record at {path}:{line_number}: {type(record).__name__}"
            )
        records.append(record)
    return tuple(records)


def _write_jsonl(path: Path, records) -> None:
    content = "\n".join(
        json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True)
        for record in records
    )
    _write_text_atomic(path, content + ("\n" if content else ""))


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
        raise NoteStoreError(f"Cannot write research records: {path}") from exc


def _safe_id(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in value
    )
