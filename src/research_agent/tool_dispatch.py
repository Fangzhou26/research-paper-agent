"""JSON tool contracts and safe dispatch for the research toolkit."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
import json
from pathlib import Path
from typing import Any, Callable, Mapping

from .toolkit import ResearchToolkit


class ToolDispatchError(ValueError):
    """Raised for invalid tool names, arguments, or registry definitions."""


@dataclass(frozen=True)
class ToolResult:
    """JSON-compatible result envelope returned to an LLM executor."""

    tool_name: str
    ok: bool
    data: Any = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "ok": self.ok,
            "data": _json_value(self.data),
            "error": self.error,
        }


@dataclass(frozen=True)
class _ToolSpec:
    name: str
    description: str
    properties: Mapping[str, Mapping[str, Any]]
    required: tuple[str, ...]
    handler: Callable[..., Any]
    tuple_fields: tuple[str, ...] = ()

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                # Optional arguments use toolkit defaults; validation remains strict
                # in the dispatcher even when provider-native strict mode is off.
                "strict": False,
                "parameters": {
                    "type": "object",
                    "properties": dict(self.properties),
                    "required": list(self.required),
                    "additionalProperties": False,
                },
            },
        }


class ResearchToolDispatcher:
    """Expose a constrained ResearchToolkit surface to one agent loop."""

    def __init__(self, toolkit: ResearchToolkit) -> None:
        self.toolkit = toolkit
        self._specs = _tool_specs(toolkit)

    @property
    def schemas(self) -> tuple[dict[str, Any], ...]:
        """Return OpenAI-compatible function tool definitions."""

        return tuple(spec.openai_schema() for spec in self._specs.values())

    @property
    def tool_names(self) -> tuple[str, ...]:
        return tuple(self._specs)

    def dispatch(
        self,
        tool_name: str,
        arguments: str | Mapping[str, Any],
    ) -> ToolResult:
        """Validate and execute one model-requested tool call."""

        spec = self._specs.get(tool_name)
        if spec is None:
            return ToolResult(tool_name, False, error=f"Unknown tool: {tool_name}")
        try:
            values = _parse_arguments(arguments)
            _validate_arguments(spec, values)
            call_values = dict(values)
            for field in spec.tuple_fields:
                if field in call_values and call_values[field] is not None:
                    call_values[field] = tuple(call_values[field])
            result = spec.handler(**call_values)
            return ToolResult(tool_name, True, data=result)
        except Exception as exc:
            return ToolResult(tool_name, False, error=f"{type(exc).__name__}: {exc}")


def _tool_specs(toolkit: ResearchToolkit) -> dict[str, _ToolSpec]:
    text = {"type": "string", "minLength": 1}
    string_array = {"type": "array", "items": text}
    positive_integer = {"type": "integer", "minimum": 1}
    specs = (
        _ToolSpec(
            "list_papers",
            "List cataloged papers available in the active research project.",
            {},
            (),
            toolkit.list_papers,
        ),
        _ToolSpec(
            "search_pages",
            "Search extracted paper pages for passages relevant to a query.",
            {
                "query": text,
                "paper_ids": string_array,
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            ("query",),
            toolkit.search_pages,
        ),
        _ToolSpec(
            "read_pages",
            "Read the full extracted text of selected one-indexed paper pages.",
            {
                "paper_id": text,
                "pages": {"type": "array", "items": positive_integer, "minItems": 1},
            },
            ("paper_id", "pages"),
            toolkit.read_pages,
        ),
        _ToolSpec(
            "add_evidence",
            "Record a page passage and a careful interpretation as project evidence.",
            {
                "paper_id": text,
                "page": positive_integer,
                "passage": text,
                "interpretation": text,
                "evidence_type": text,
                "support_type": {"type": "string", "enum": ["direct", "inference"]},
            },
            ("paper_id", "page", "passage", "interpretation", "evidence_type"),
            toolkit.add_evidence,
        ),
        _ToolSpec(
            "list_evidence",
            "List recorded evidence, optionally restricted to one paper.",
            {"paper_id": {"type": ["string", "null"], "minLength": 1}},
            (),
            toolkit.list_evidence,
        ),
        _ToolSpec(
            "save_note",
            "Save a structured, evidence-linked analysis note for one paper.",
            {
                "paper_id": text,
                "research_problem": text,
                "method": text,
                "data_or_inputs": string_array,
                "key_contributions": string_array,
                "main_results": string_array,
                "limitations": string_array,
                "relevance": {"type": ["string", "null"]},
                "topics": string_array,
                "evidence_ids": string_array,
                "uncertainties": string_array,
            },
            ("paper_id", "research_problem", "method"),
            toolkit.save_note,
            (
                "data_or_inputs", "key_contributions", "main_results", "limitations",
                "topics", "evidence_ids", "uncertainties",
            ),
        ),
        _ToolSpec(
            "load_note",
            "Load the current structured note for a paper, if one exists.",
            {"paper_id": text},
            ("paper_id",),
            toolkit.load_note,
        ),
        _ToolSpec(
            "citation_key",
            "Get the verified bibliography citation key for a cataloged paper.",
            {"paper_id": text},
            ("paper_id",),
            toolkit.citation_key,
        ),
        _ToolSpec(
            "audit_citations",
            "Check LaTeX citation keys in draft text against the project bibliography.",
            {"text": {"type": "string"}},
            ("text",),
            toolkit.audit_citations,
        ),
        _ToolSpec(
            "read_output",
            "Read a provenance-verified generated output by artifact ID.",
            {"artifact_id": text},
            ("artifact_id",),
            toolkit.read_output,
        ),
        _ToolSpec(
            "write_output",
            "Write a final project artifact with paper and evidence provenance.",
            {
                "filename": text,
                "content": {"type": "string"},
                "paper_ids": string_array,
                "evidence_ids": string_array,
            },
            ("filename", "content"),
            toolkit.write_output,
            ("paper_ids", "evidence_ids"),
        ),
    )
    return {spec.name: spec for spec in specs}


def _parse_arguments(arguments: str | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ToolDispatchError("Tool arguments are not valid JSON") from exc
    else:
        value = arguments
    if not isinstance(value, Mapping):
        raise ToolDispatchError("Tool arguments must be a JSON object")
    return dict(value)


def _validate_arguments(spec: _ToolSpec, values: Mapping[str, Any]) -> None:
    unknown = set(values) - set(spec.properties)
    if unknown:
        raise ToolDispatchError(f"Unknown arguments: {', '.join(sorted(unknown))}")
    missing = set(spec.required) - set(values)
    if missing:
        raise ToolDispatchError(f"Missing arguments: {', '.join(sorted(missing))}")
    for name, value in values.items():
        _validate_value(name, value, spec.properties[name])


def _validate_value(name: str, value: Any, schema: Mapping[str, Any]) -> None:
    expected = schema.get("type")
    allowed_types = (expected,) if isinstance(expected, str) else tuple(expected or ())
    if value is None and "null" in allowed_types:
        return
    matches = (
        ("string" in allowed_types and isinstance(value, str))
        or ("integer" in allowed_types and isinstance(value, int) and not isinstance(value, bool))
        or ("boolean" in allowed_types and isinstance(value, bool))
        or ("array" in allowed_types and isinstance(value, list))
    )
    if not matches:
        raise ToolDispatchError(f"Argument '{name}' has the wrong type")
    if isinstance(value, str):
        if schema.get("minLength", 0) > len(value):
            raise ToolDispatchError(f"Argument '{name}' must not be empty")
        if "enum" in schema and value not in schema["enum"]:
            raise ToolDispatchError(f"Argument '{name}' has an unsupported value")
    if isinstance(value, int) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            raise ToolDispatchError(f"Argument '{name}' is below the minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise ToolDispatchError(f"Argument '{name}' is above the maximum")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0):
            raise ToolDispatchError(f"Argument '{name}' has too few items")
        item_schema = schema.get("items")
        if item_schema:
            for index, item in enumerate(value):
                _validate_value(f"{name}[{index}]", item, item_schema)


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value
