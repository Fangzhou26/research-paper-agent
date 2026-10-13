"""File-backed catalog for canonical paper records."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile

from .models import PaperRecord, SchemaValidationError, record_from_dict, record_to_dict


class CatalogError(RuntimeError):
    """Raised when a paper catalog cannot be read or written safely."""


class CatalogConflictError(CatalogError):
    """Raised when one paper ID is associated with different records."""


@dataclass(frozen=True)
class CatalogAddResult:
    """Result of adding a paper record to a catalog."""

    record: PaperRecord
    created: bool


class PaperCatalog:
    """A small JSONL catalog with deterministic, conflict-safe writes."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def all(self) -> tuple[PaperRecord, ...]:
        """Load all records in catalog order."""

        if not self.path.exists():
            return ()
        records: list[PaperRecord] = []
        paper_ids: set[str] = set()
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            raise CatalogError(f"Cannot read paper catalog: {self.path}") from exc
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                serialized = json.loads(line)
                record = record_from_dict(serialized)
            except (json.JSONDecodeError, SchemaValidationError) as exc:
                raise CatalogError(
                    f"Invalid catalog record at {self.path}:{line_number}"
                ) from exc
            if not isinstance(record, PaperRecord):
                raise CatalogError(
                    f"Non-paper record at {self.path}:{line_number}: "
                    f"{type(record).__name__}"
                )
            if record.paper_id in paper_ids:
                raise CatalogError(
                    f"Duplicate paper ID at {self.path}:{line_number}: {record.paper_id}"
                )
            paper_ids.add(record.paper_id)
            records.append(record)
        return tuple(records)

    def get(self, paper_id: str) -> PaperRecord | None:
        """Return a paper by stable ID, or `None` when absent."""

        return next(
            (record for record in self.all() if record.paper_id == paper_id),
            None,
        )

    def add(self, record: PaperRecord) -> CatalogAddResult:
        """Add a record without overwriting conflicting metadata."""

        records = list(self.all())
        existing = next(
            (item for item in records if item.paper_id == record.paper_id),
            None,
        )
        if existing is not None:
            if existing != record:
                raise CatalogConflictError(
                    f"Paper ID already has different metadata: {record.paper_id}"
                )
            return CatalogAddResult(record=existing, created=False)
        records.append(record)
        self._write(records)
        return CatalogAddResult(record=record, created=True)

    def _write(self, records: list[PaperRecord]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            json.dumps(record_to_dict(record), ensure_ascii=False, sort_keys=True)
            for record in records
        ]
        content = "\n".join(lines) + ("\n" if lines else "")
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=f".{self.path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, self.path)
        except OSError as exc:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise CatalogError(f"Cannot write paper catalog: {self.path}") from exc

