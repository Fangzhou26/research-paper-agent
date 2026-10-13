"""Versioned data contracts shared by research tools and skills."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, TypeAlias


SCHEMA_VERSION = 1


class SchemaValidationError(ValueError):
    """Raised when a research record violates its data contract."""


def _required_text(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise SchemaValidationError(f"'{field_name}' must be a non-empty string")


def _optional_text(value: str | None, field_name: str) -> None:
    if value is not None and not isinstance(value, str):
        raise SchemaValidationError(f"'{field_name}' must be a string or null")


def _string_tuple(value: tuple[str, ...], field_name: str) -> None:
    if not isinstance(value, tuple) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise SchemaValidationError(
            f"'{field_name}' must be a tuple of non-empty strings"
        )


@dataclass(frozen=True)
class PaperRecord:
    """Canonical identity, source, and bibliographic facts for one paper."""

    paper_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    source_type: str = "uploaded"
    source_uri: str | None = None
    file_path: Path | None = None
    verification_status: str = "unverified"
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.paper_id, "paper_id")
        _required_text(self.title, "title")
        _string_tuple(self.authors, "authors")
        _string_tuple(self.tags, "tags")
        _required_text(self.source_type, "source_type")
        _required_text(self.verification_status, "verification_status")
        for field_name in ("venue", "doi", "arxiv_id", "url", "source_uri"):
            _optional_text(getattr(self, field_name), field_name)
        if self.year is not None and (
            not isinstance(self.year, int) or isinstance(self.year, bool) or self.year < 0
        ):
            raise SchemaValidationError("'year' must be a non-negative integer or null")
        if self.file_path is not None and not isinstance(self.file_path, Path):
            raise SchemaValidationError("'file_path' must be a Path or null")


@dataclass(frozen=True)
class PageText:
    """Text extracted from one page of a paper."""

    paper_id: str
    page: int
    text: str
    extraction_method: str
    quality: str = "unknown"

    def __post_init__(self) -> None:
        _required_text(self.paper_id, "paper_id")
        _required_text(self.extraction_method, "extraction_method")
        _required_text(self.quality, "quality")
        if not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1:
            raise SchemaValidationError("'page' must be an integer greater than zero")
        if not isinstance(self.text, str):
            raise SchemaValidationError("'text' must be a string")


@dataclass(frozen=True)
class EvidenceRecord:
    """A page-level passage and the interpretation derived from it."""

    evidence_id: str
    project_id: str
    paper_id: str
    page: int
    passage: str
    interpretation: str
    evidence_type: str
    support_type: str = "direct"

    def __post_init__(self) -> None:
        for field_name in (
            "evidence_id",
            "project_id",
            "paper_id",
            "passage",
            "interpretation",
            "evidence_type",
            "support_type",
        ):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.page, int) or isinstance(self.page, bool) or self.page < 1:
            raise SchemaValidationError("'page' must be an integer greater than zero")


@dataclass(frozen=True)
class PaperNote:
    """Project-specific structured analysis of one paper."""

    project_id: str
    paper_id: str
    research_problem: str
    method: str
    data_or_inputs: tuple[str, ...] = ()
    key_contributions: tuple[str, ...] = ()
    main_results: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    relevance: str | None = None
    topics: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    uncertainties: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("project_id", "paper_id", "research_problem", "method"):
            _required_text(getattr(self, field_name), field_name)
        _optional_text(self.relevance, "relevance")
        for field_name in (
            "data_or_inputs",
            "key_contributions",
            "main_results",
            "limitations",
            "topics",
            "evidence_ids",
            "uncertainties",
        ):
            _string_tuple(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class SearchCandidate:
    """A discovered source that has not necessarily entered the paper catalog."""

    candidate_id: str
    title: str
    authors: tuple[str, ...] = ()
    year: int | None = None
    source_url: str | None = None
    external_id: str | None = None
    abstract: str | None = None
    doi: str | None = None
    venue: str | None = None
    cited_by_count: int = 0
    relevance_score: float = 0.0
    discovery_sources: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()
    screening_decision: str = "unreviewed"
    screening_reason: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.candidate_id, "candidate_id")
        _required_text(self.title, "title")
        for field_name in ("authors", "discovery_sources", "topics"):
            _string_tuple(getattr(self, field_name), field_name)
        _required_text(self.screening_decision, "screening_decision")
        for field_name in (
            "source_url", "external_id", "abstract", "doi", "venue",
            "screening_reason",
        ):
            _optional_text(getattr(self, field_name), field_name)
        if self.year is not None and (
            not isinstance(self.year, int) or isinstance(self.year, bool) or self.year < 0
        ):
            raise SchemaValidationError("'year' must be a non-negative integer or null")
        if (
            not isinstance(self.cited_by_count, int)
            or isinstance(self.cited_by_count, bool)
            or self.cited_by_count < 0
        ):
            raise SchemaValidationError("'cited_by_count' must be a non-negative integer")
        if not isinstance(self.relevance_score, (int, float)) or isinstance(
            self.relevance_score, bool
        ):
            raise SchemaValidationError("'relevance_score' must be a number")


@dataclass(frozen=True)
class InnovationProfile:
    """Run input describing the new work and its seed literature."""

    project_id: str
    topic: str
    innovation_points: tuple[str, ...]
    seed_paper_ids: tuple[str, ...]
    constraints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _required_text(self.project_id, "project_id")
        _required_text(self.topic, "topic")
        for field_name in ("innovation_points", "seed_paper_ids", "constraints"):
            _string_tuple(getattr(self, field_name), field_name)
        if not 1 <= len(self.innovation_points) <= 10:
            raise SchemaValidationError("'innovation_points' must contain 1 to 10 items")
        if not 1 <= len(self.seed_paper_ids) <= 3:
            raise SchemaValidationError("'seed_paper_ids' must contain 1 to 3 items")


@dataclass(frozen=True)
class BibliographyEntry:
    """A citation key associated with verified bibliographic metadata."""

    citation_key: str
    paper_id: str
    entry_type: str
    fields: tuple[tuple[str, str], ...]
    verification_status: str = "unverified"

    def __post_init__(self) -> None:
        _required_text(self.citation_key, "citation_key")
        _required_text(self.paper_id, "paper_id")
        _required_text(self.entry_type, "entry_type")
        _required_text(self.verification_status, "verification_status")
        if not isinstance(self.fields, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not all(isinstance(part, str) and part.strip() for part in item)
            for item in self.fields
        ):
            raise SchemaValidationError(
                "'fields' must be a tuple of non-empty string pairs"
            )


@dataclass(frozen=True)
class ResearchTask:
    """One dependency-aware unit in a decomposed research plan."""

    task_id: str
    question: str
    task_type: str
    retrieval_queries: tuple[str, ...] = ()
    paper_ids: tuple[str, ...] = ()
    required_evidence_types: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    status: str = "pending"

    def __post_init__(self) -> None:
        for field_name in ("task_id", "question", "task_type", "status"):
            _required_text(getattr(self, field_name), field_name)
        for field_name in (
            "retrieval_queries",
            "paper_ids",
            "required_evidence_types",
            "dependencies",
        ):
            _string_tuple(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class ResearchPlan:
    """A validated decomposition from research objective to deliverables."""

    plan_id: str
    project_id: str
    objective: str
    assumptions: tuple[str, ...] = ()
    subquestions: tuple[str, ...] = ()
    tasks: tuple[ResearchTask, ...] = ()
    deliverables: tuple[str, ...] = ()
    completion_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for field_name in ("plan_id", "project_id", "objective"):
            _required_text(getattr(self, field_name), field_name)
        for field_name in (
            "assumptions",
            "subquestions",
            "deliverables",
            "completion_criteria",
        ):
            _string_tuple(getattr(self, field_name), field_name)
        if not isinstance(self.tasks, tuple) or any(
            not isinstance(task, ResearchTask) for task in self.tasks
        ):
            raise SchemaValidationError(
                "'tasks' must be a tuple of ResearchTask records"
            )


@dataclass(frozen=True)
class RunState:
    """Minimal resumable state for one research task."""

    task_id: str
    project_id: str
    task_type: str
    status: str
    current_stage: str
    completed_steps: tuple[str, ...] = ()
    artifact_ids: tuple[str, ...] = ()
    open_questions: tuple[str, ...] = ()
    summary: str | None = None
    updated_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "project_id",
            "task_type",
            "status",
            "current_stage",
        ):
            _required_text(getattr(self, field_name), field_name)
        for field_name in ("completed_steps", "artifact_ids", "open_questions"):
            _string_tuple(getattr(self, field_name), field_name)
        _optional_text(self.summary, "summary")
        _optional_text(self.updated_at, "updated_at")


@dataclass(frozen=True)
class ArtifactRecord:
    """A generated output and the records used to produce it."""

    artifact_id: str
    project_id: str
    artifact_type: str
    path: Path
    paper_ids: tuple[str, ...] = ()
    evidence_ids: tuple[str, ...] = ()
    source_artifact_ids: tuple[str, ...] = ()
    created_at: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("artifact_id", "project_id", "artifact_type"):
            _required_text(getattr(self, field_name), field_name)
        if not isinstance(self.path, Path):
            raise SchemaValidationError("'path' must be a Path")
        for field_name in ("paper_ids", "evidence_ids", "source_artifact_ids"):
            _string_tuple(getattr(self, field_name), field_name)
        _optional_text(self.created_at, "created_at")


ResearchRecord: TypeAlias = (
    PaperRecord
    | PageText
    | EvidenceRecord
    | PaperNote
    | SearchCandidate
    | InnovationProfile
    | BibliographyEntry
    | ResearchTask
    | ResearchPlan
    | RunState
    | ArtifactRecord
)


_RECORD_TYPES: dict[str, type[ResearchRecord]] = {
    record_type.__name__: record_type
    for record_type in (
        PaperRecord,
        PageText,
        EvidenceRecord,
        PaperNote,
        SearchCandidate,
        InnovationProfile,
        BibliographyEntry,
        ResearchTask,
        ResearchPlan,
        RunState,
        ArtifactRecord,
    )
}

_PATH_FIELDS: dict[str, set[str]] = {
    "PaperRecord": {"file_path"},
    "ArtifactRecord": {"path"},
}

_TUPLE_FIELDS: dict[str, set[str]] = {
    name: {
        field.name
        for field in fields(record_type)
        if "tuple" in str(field.type).lower()
    }
    for name, record_type in _RECORD_TYPES.items()
}


def record_to_dict(record: ResearchRecord) -> dict[str, Any]:
    """Convert a research record to a versioned JSON-compatible mapping."""

    if not is_dataclass(record) or record.__class__.__name__ not in _RECORD_TYPES:
        raise TypeError(f"Unsupported research record: {type(record).__name__}")
    data = _json_value(asdict(record))
    return {
        "schema_version": SCHEMA_VERSION,
        "record_type": record.__class__.__name__,
        **data,
    }


def record_from_dict(data: Mapping[str, Any]) -> ResearchRecord:
    """Construct and validate a research record from a serialized mapping."""

    if not isinstance(data, Mapping):
        raise SchemaValidationError("Serialized record must be an object")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise SchemaValidationError(
            f"Unsupported schema version: {data.get('schema_version')!r}"
        )
    record_name = data.get("record_type")
    record_type = _RECORD_TYPES.get(record_name) if isinstance(record_name, str) else None
    if record_type is None:
        raise SchemaValidationError(f"Unknown record type: {record_name!r}")

    values = {
        key: value
        for key, value in data.items()
        if key not in {"schema_version", "record_type"}
    }
    allowed_fields = {field.name for field in fields(record_type)}
    unknown_fields = set(values) - allowed_fields
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise SchemaValidationError(f"Unknown fields for {record_name}: {unknown}")
    for field_name in _PATH_FIELDS.get(record_name, set()):
        if field_name in values and values[field_name] is not None:
            values[field_name] = Path(values[field_name])
    for field_name in _TUPLE_FIELDS.get(record_name, set()):
        if field_name in values:
            value = values[field_name]
            if not isinstance(value, list):
                raise SchemaValidationError(
                    f"Serialized '{field_name}' must be a JSON array"
                )
            if record_name == "BibliographyEntry" and field_name == "fields":
                values[field_name] = tuple(tuple(item) for item in value)
            elif record_name == "ResearchPlan" and field_name == "tasks":
                values[field_name] = tuple(
                    _research_task_from_mapping(item) for item in value
                )
            else:
                values[field_name] = tuple(value)
    try:
        return record_type(**values)
    except TypeError as exc:
        raise SchemaValidationError(f"Invalid {record_name}: {exc}") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _research_task_from_mapping(value: Any) -> ResearchTask:
    if not isinstance(value, Mapping):
        raise SchemaValidationError("Serialized plan task must be an object")
    task_values = dict(value)
    allowed_fields = {field.name for field in fields(ResearchTask)}
    unknown_fields = set(task_values) - allowed_fields
    if unknown_fields:
        unknown = ", ".join(sorted(unknown_fields))
        raise SchemaValidationError(f"Unknown fields for ResearchTask: {unknown}")
    for field_name in _TUPLE_FIELDS["ResearchTask"]:
        if field_name in task_values:
            item = task_values[field_name]
            if not isinstance(item, list):
                raise SchemaValidationError(
                    f"Serialized ResearchTask '{field_name}' must be a JSON array"
                )
            task_values[field_name] = tuple(item)
    try:
        return ResearchTask(**task_values)
    except TypeError as exc:
        raise SchemaValidationError(f"Invalid ResearchTask: {exc}") from exc
