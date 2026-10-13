import json
from dataclasses import replace
from pathlib import Path

from research_agent import (
    BibliographyStore,
    EvidenceStore,
    OutputWriter,
    PaperNoteStore,
    ResearchContext,
    ResearchToolkit,
    ResearchToolDispatcher,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def dispatcher(tmp_path: Path) -> ResearchToolDispatcher:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    context = replace(
        context,
        project=replace(context.project, output_directory=tmp_path / "outputs"),
    )
    evidence_store = EvidenceStore(context, tmp_path / "evidence.jsonl")
    note_store = PaperNoteStore(context, tmp_path / "notes", evidence_store=evidence_store)
    bibliography_store = BibliographyStore(context)
    output_writer = OutputWriter(
        context,
        evidence_store=evidence_store,
        bibliography_store=bibliography_store,
    )
    toolkit = ResearchToolkit(
        context,
        evidence_store=evidence_store,
        note_store=note_store,
        bibliography_store=bibliography_store,
        output_writer=output_writer,
    )
    return ResearchToolDispatcher(toolkit)


def test_schemas_are_openai_function_tools_with_closed_arguments(tmp_path: Path) -> None:
    tools = dispatcher(tmp_path)

    assert len(tools.schemas) == 11
    assert "search_pages" in tools.tool_names
    assert "read_output" in tools.tool_names
    for schema in tools.schemas:
        assert schema["type"] == "function"
        assert schema["function"]["strict"] is False
        assert schema["function"]["parameters"]["additionalProperties"] is False

    by_name = {schema["function"]["name"]: schema for schema in tools.schemas}
    assert "overwrite" not in by_name["save_note"]["function"]["parameters"]["properties"]
    assert "overwrite" not in by_name["write_output"]["function"]["parameters"]["properties"]


def test_dispatches_json_arguments_and_serializes_dataclass_results(tmp_path: Path) -> None:
    tools = dispatcher(tmp_path)

    result = tools.dispatch("search_pages", json.dumps({"query": "synthetic sample study", "limit": 1}))
    payload = result.to_dict()

    assert result.ok is True
    assert len(payload["data"]) == 1
    assert payload["data"][0]["paper_id"]
    assert isinstance(payload["data"][0]["page"], int)


def test_dispatcher_rejects_unknown_missing_and_extra_arguments(tmp_path: Path) -> None:
    tools = dispatcher(tmp_path)

    assert tools.dispatch("not_a_tool", {}).error == "Unknown tool: not_a_tool"
    missing = tools.dispatch("read_pages", {"paper_id": "paper"})
    extra = tools.dispatch("list_papers", {"repository_path": "/tmp"})

    assert missing.ok is False
    assert "Missing arguments: pages" in missing.error
    assert extra.ok is False
    assert "Unknown arguments: repository_path" in extra.error


def test_dispatcher_validates_nested_types_before_call(tmp_path: Path) -> None:
    tools = dispatcher(tmp_path)

    result = tools.dispatch("read_pages", {"paper_id": "paper", "pages": [True]})

    assert result.ok is False
    assert "wrong type" in result.error


def test_dispatcher_returns_tool_failures_without_raising(tmp_path: Path) -> None:
    tools = dispatcher(tmp_path)

    result = tools.dispatch("citation_key", {"paper_id": "unknown-paper"})

    assert result.ok is False
    assert result.error.startswith("BibliographyError:")
