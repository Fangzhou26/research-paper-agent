import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from research_agent import (
    EvidenceStore,
    LLMAssistantTurn,
    LLMToolCall,
    ResearchContext,
    ResearchTask,
    ResearchTaskExecutor,
    ResearchToolkit,
    ResearchToolDispatcher,
    PaperNoteStore,
    RunStateStore,
    RunState,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StubToolProvider:
    def __init__(self, turns: list[LLMAssistantTurn]) -> None:
        self.turns = turns
        self.calls: list[tuple[Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]]]] = []

    def complete_with_tools(self, messages, tools) -> LLMAssistantTurn:
        self.calls.append((tuple(messages), tuple(tools)))
        return self.turns.pop(0)


def tool_turn(call_id: str, name: str, arguments: Mapping[str, Any]) -> LLMAssistantTurn:
    return LLMAssistantTurn(
        None,
        (LLMToolCall(call_id, name, json.dumps(arguments)),),
        "tool_calls",
    )


def executor(
    tmp_path: Path,
    provider: StubToolProvider,
    *,
    max_rounds: int = 4,
    on_event=None,
) -> ResearchTaskExecutor:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    dispatcher = ResearchToolDispatcher(ResearchToolkit(context))
    return ResearchTaskExecutor(
        context,
        provider,
        dispatcher,
        state_store=RunStateStore(context, tmp_path / "runs"),
        max_rounds=max_rounds,
        on_event=on_event,
    )


def overview_task() -> ResearchTask:
    return ResearchTask(
        task_id="overview-loop",
        question="What papers are in the corpus?",
        task_type="corpus_overview",
    )


def test_executor_calls_allowed_tool_then_finishes_and_checkpoints(tmp_path: Path) -> None:
    events: list[tuple[str, Mapping[str, Any]]] = []
    provider = StubToolProvider([
        tool_turn("call-1", "list_papers", {}),
        tool_turn("call-2", "finish_task", {
            "status": "completed",
            "summary": "The corpus was inspected.",
            "open_questions": [],
        }),
    ])
    runner = executor(
        tmp_path,
        provider,
        on_event=lambda event_type, data: events.append((event_type, data)),
    )

    result = runner.execute(overview_task(), completion_criteria=("Corpus inspected",))

    assert result.state.status == "completed"
    assert result.state.completed_steps == ("list_papers:ok",)
    assert result.summary == "The corpus was inspected."
    assert result.state.summary == "The corpus was inspected."
    assert runner.state_store.load(overview_task().task_id) == result.state
    offered = {schema["function"]["name"] for schema in provider.calls[0][1]}
    assert offered == {
        "list_papers", "search_pages", "read_pages", "load_note",
        "list_evidence", "finish_task",
    }
    second_messages = provider.calls[1][0]
    assert any(message["role"] == "tool" for message in second_messages)
    assert [event_type for event_type, _ in events] == [
        "task_started",
        "tool_requested",
        "tool_succeeded",
        "tool_requested",
        "tool_succeeded",
        "task_finishing",
    ]


def test_plain_text_does_not_finish_task(tmp_path: Path) -> None:
    provider = StubToolProvider([
        LLMAssistantTurn("I am done.", (), "stop"),
        tool_turn("call-2", "finish_task", {
            "status": "needs_input",
            "summary": "A scope choice is required.",
            "open_questions": ["Which papers should be included?"],
        }),
    ])

    result = executor(tmp_path, provider).execute(overview_task())

    assert result.state.status == "needs_input"
    assert result.state.open_questions == ("Which papers should be included?",)
    assert "plain text does not finish" in provider.calls[1][0][-1]["content"]


def test_executor_blocks_after_maximum_rounds(tmp_path: Path) -> None:
    provider = StubToolProvider([LLMAssistantTurn("thinking", (), "stop")])

    result = executor(tmp_path, provider, max_rounds=1).execute(overview_task())

    assert result.state.status == "blocked"
    assert result.state.current_stage == "max_rounds_exceeded"


def test_terminal_checkpoint_prevents_duplicate_execution(tmp_path: Path) -> None:
    provider = StubToolProvider([tool_turn("call-1", "finish_task", {
        "status": "completed",
        "summary": "Complete.",
        "open_questions": [],
    })])
    runner = executor(tmp_path, provider)
    runner.execute(overview_task())

    repeated = runner.execute(overview_task())

    assert repeated.rounds == 0
    assert repeated.summary == "Complete."
    assert provider.turns == []


def test_blocked_checkpoint_retries_only_when_explicit(tmp_path: Path) -> None:
    provider = StubToolProvider([tool_turn("retry", "finish_task", {
        "status": "completed",
        "summary": "Completed after capability repair.",
        "open_questions": [],
    })])
    runner = executor(tmp_path, provider)
    runner.state_store.save(RunState(
        task_id=overview_task().task_id,
        project_id="demo",
        task_type="corpus_overview",
        status="blocked",
        current_stage="finished",
        summary="Missing tool.",
    ))

    unchanged = runner.execute(overview_task())
    retried = runner.execute(overview_task(), retry_blocked=True)

    assert unchanged.state.status == "blocked"
    assert retried.state.status == "completed"
    assert len(provider.calls) == 1


def test_executor_resumes_persisted_tool_conversation(tmp_path: Path) -> None:
    first_provider = StubToolProvider([LLMAssistantTurn("Inspecting first.", (), "stop")])
    runner = executor(tmp_path, first_provider, max_rounds=1)
    first = runner.execute(overview_task())
    assert first.state.status == "blocked"

    second_provider = StubToolProvider([tool_turn("done", "finish_task", {
        "status": "completed",
        "summary": "Continued from the prior observation.",
        "open_questions": [],
    })])
    resumed = ResearchTaskExecutor(
        runner.context,
        second_provider,
        runner.dispatcher,
        state_store=runner.state_store,
        max_rounds=1,
    ).execute(overview_task(), retry_blocked=True)

    assert resumed.state.status == "completed"
    resumed_messages = second_provider.calls[0][0]
    assert any(message.get("content") == "Inspecting first." for message in resumed_messages)
    assert any("Retry this blocked task" in str(message.get("content")) for message in resumed_messages)


def test_invalid_finish_call_is_returned_to_model_for_repair(tmp_path: Path) -> None:
    provider = StubToolProvider([
        tool_turn("bad", "finish_task", {"status": "completed"}),
        tool_turn("good", "finish_task", {
            "status": "completed",
            "summary": "Complete after correction.",
            "open_questions": [],
        }),
    ])

    result = executor(tmp_path, provider).execute(overview_task())

    assert result.state.status == "completed"
    assert "finish_task:error" in result.state.completed_steps
    error_message = provider.calls[1][0][-1]
    assert error_message["role"] == "tool"
    assert json.loads(error_message["content"])["ok"] is False


def test_paper_analysis_cannot_complete_without_note_and_evidence(tmp_path: Path) -> None:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    evidence_store = EvidenceStore(context, tmp_path / "empty-evidence.jsonl")
    note_store = PaperNoteStore(
        context,
        tmp_path / "empty-notes",
        evidence_store=evidence_store,
    )
    tools = ResearchToolkit(
        context,
        evidence_store=evidence_store,
        note_store=note_store,
    )
    paper_id = tools.list_papers()[0].paper_id
    task = ResearchTask(
        task_id="analysis-guard",
        question="Analyze the method and its limitations.",
        task_type="paper_analysis",
        retrieval_queries=("method limitations",),
        paper_ids=(paper_id,),
        required_evidence_types=("method", "limitation"),
    )
    provider = StubToolProvider([
        tool_turn("premature", "finish_task", {
            "status": "completed",
            "summary": "Done without evidence.",
            "open_questions": [],
        }),
        tool_turn("blocked", "finish_task", {
            "status": "needs_input",
            "summary": "Evidence must be collected first.",
            "open_questions": ["Continue evidence collection?"],
        }),
    ])

    runner = ResearchTaskExecutor(
        context,
        provider,
        ResearchToolDispatcher(tools),
        state_store=RunStateStore(context, tmp_path / "runs"),
    )
    result = runner.execute(task)

    assert result.state.status == "needs_input"
    completion_error = json.loads(provider.calls[1][0][-1]["content"])["error"]
    assert "paper note is missing" in completion_error
    assert "required evidence type is missing: method" in completion_error


def test_non_analysis_task_does_not_require_writable_evidence_types(tmp_path: Path) -> None:
    task = ResearchTask(
        task_id="overview-with-guidance",
        question="Inspect abstracts and introductions.",
        task_type="corpus_overview",
        required_evidence_types=("abstracts", "introduction sections"),
    )
    provider = StubToolProvider([tool_turn("done", "finish_task", {
        "status": "completed",
        "summary": "The corpus overview is complete.",
        "open_questions": [],
    })])

    result = executor(tmp_path, provider).execute(task)

    assert result.state.status == "completed"
