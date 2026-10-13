"""Deterministic offline demonstration of the production research-agent loop."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from .execution import ResearchTaskExecutor, RunStateStore
from .llm import LLMAssistantTurn, LLMProviderError, LLMToolCall
from .models import ResearchPlan, ResearchTask, record_to_dict
from .plan_execution import ResearchPlanExecutor
from .planning import ResearchPlanStore, validate_agent_plan
from .project import ResearchContext
from .tool_dispatch import ResearchToolDispatcher
from .toolkit import ResearchToolkit


class DemoError(RuntimeError):
    """Raised when the selected project cannot support the offline demo."""


DemoEventSink = Callable[[str, Mapping[str, Any]], None]


@dataclass(frozen=True)
class DemoResult:
    """Inspectable outcome of one offline demonstration."""

    workspace: Path
    plan_path: Path
    plan_id: str
    planned_task_ids: tuple[str, ...]
    first_attempt_status: str
    first_attempt_stopped_task: str | None
    resumed_status: str
    completed_task_ids: tuple[str, ...]
    unsafe_citation_keys: tuple[str, ...]
    artifact_path: Path
    manifest_path: Path
    tool_trace: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        failure_injected = self.first_attempt_status == "blocked"
        return {
            "demo": "evidence-grounded-research-agent",
            "mode": "offline_scripted_provider",
            "workspace": str(self.workspace),
            "plan": {
                "validation": "passed",
                "plan_id": self.plan_id,
                "task_ids": list(self.planned_task_ids),
                "path": str(self.plan_path),
            },
            "failure": {
                "injected": failure_injected,
                "first_attempt_status": self.first_attempt_status,
                "stopped_task_id": self.first_attempt_stopped_task,
                "reason": "simulated provider outage" if failure_injected else None,
            },
            "recovery": {
                "action": "retry_blocked" if failure_injected else "not_required",
                "status": self.resumed_status,
                "completed_task_ids": list(self.completed_task_ids),
            },
            "citation_guard": {
                "candidate_rejected": bool(self.unsafe_citation_keys),
                "unknown_keys": list(self.unsafe_citation_keys),
            },
            "output": {
                "artifact_path": str(self.artifact_path),
                "manifest_path": str(self.manifest_path),
            },
            "tool_trace": list(self.tool_trace),
        }


class DemoProvider:
    """Script model decisions while exercising the real task loop and tools."""

    def __init__(
        self,
        paper_id: str,
        evidence_id: str,
        *,
        fail_once: bool = True,
        on_event: DemoEventSink | None = None,
        step_delay: float = 0,
    ) -> None:
        self.paper_id = paper_id
        self.evidence_id = evidence_id
        self.fail_once = fail_once
        self.failed = False
        self.tool_trace: list[str] = []
        self.unsafe_citation_keys: tuple[str, ...] = ()
        self.on_event = on_event
        self.step_delay = max(0, step_delay)
        self._seen_tool_payloads: dict[str, int] = {}
        self._started_tasks: set[str] = set()

    def complete_with_tools(
        self,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> LLMAssistantTurn:
        task_prompt = _task_prompt(messages)
        task_id = task_prompt["task"]["task_id"]
        if task_id not in self._started_tasks:
            self._started_tasks.add(task_id)
            self._emit("task_started", {
                "task_id": task_id,
                "task_type": task_prompt["task"]["task_type"],
                "question": task_prompt["task"]["question"],
            })
        if self.fail_once and not self.failed:
            self.failed = True
            self._emit("provider_failed", {
                "task_id": task_id,
                "error_code": "provider_unavailable",
                "message": "Demo fault injection: provider temporarily unavailable",
                "retryable": True,
            })
            raise LLMProviderError(
                "Demo fault injection: provider temporarily unavailable"
            )

        tool_payloads = _tool_payloads(messages)
        self._emit_new_observations(tool_payloads, task_id)
        if task_id == "demo-analyze":
            return self._analysis_turn(tool_payloads)
        if task_id == "demo-synthesize":
            return self._synthesis_turn(tool_payloads)
        if task_id == "demo-audit":
            return self._audit_turn(tool_payloads, task_prompt)
        raise DemoError(f"Unexpected demo task: {task_id}")

    def _analysis_turn(self, payloads: list[Mapping[str, Any]]) -> LLMAssistantTurn:
        if not payloads:
            return self._call("search_pages", {
                "query": "synthetic sample study",
                "paper_ids": [self.paper_id],
                "limit": 1,
            })
        last = payloads[-1]
        if last["tool_name"] == "search_pages":
            matches = last.get("data") or []
            if not matches:
                raise DemoError("Demo query returned no extracted page")
            return self._call("read_pages", {
                "paper_id": self.paper_id,
                "pages": [matches[0]["page"]],
            })
        if last["tool_name"] == "read_pages":
            return self._call("load_note", {"paper_id": self.paper_id})
        if last["tool_name"] == "load_note":
            return self._call("list_evidence", {"paper_id": self.paper_id})
        return self._finish(
            "Analysis reused a structured note and page-level evidence."
        )

    def _synthesis_turn(self, payloads: list[Mapping[str, Any]]) -> LLMAssistantTurn:
        if not payloads:
            return self._call("load_note", {"paper_id": self.paper_id})
        last = payloads[-1]
        if last["tool_name"] == "load_note":
            return self._call("citation_key", {"paper_id": self.paper_id})
        if last["tool_name"] == "citation_key":
            key = last["data"]
            content = (
                "# Evidence-backed synthesis\n\n"
                "The selected synthetic study demonstrates page-linked evidence "
                f"and records its method and limitations as structured evidence \\cite{{{key}}}.\n"
            )
            return self._call("write_output", {
                "filename": "demo_synthesis.md",
                "content": content,
                "paper_ids": [self.paper_id],
                "evidence_ids": [self.evidence_id],
            })
        return self._finish(
            "A citation-safe synthesis artifact was written with provenance."
        )

    def _audit_turn(
        self,
        payloads: list[Mapping[str, Any]],
        task_prompt: Mapping[str, Any],
    ) -> LLMAssistantTurn:
        if not payloads:
            return self._call("audit_citations", {
                "text": r"Unsafe candidate \cite{invented2026}.",
            })
        last = payloads[-1]
        if (
            last["tool_name"] == "audit_citations"
            and last["data"]["unknown_keys"]
        ):
            self.unsafe_citation_keys = tuple(last["data"]["unknown_keys"])
            self._emit("citation_rejected", {
                "task_id": "demo-audit",
                "unknown_keys": list(self.unsafe_citation_keys),
            })
            return self._call("read_output", {
                "artifact_id": _dependency_artifact_id(task_prompt),
            })
        if last["tool_name"] == "read_output":
            return self._call("audit_citations", {
                "text": last["data"]["content"],
            })
        return self._finish("The final artifact passed citation-key validation.")

    def _call(self, name: str, arguments: Mapping[str, Any]) -> LLMAssistantTurn:
        self.tool_trace.append(name)
        self._emit("tool_requested", {
            "tool_name": name,
            "arguments": dict(arguments),
        })
        return LLMAssistantTurn(
            None,
            (LLMToolCall(
                f"demo-call-{len(self.tool_trace)}",
                name,
                json.dumps(arguments),
            ),),
            "tool_calls",
        )

    def _finish(self, summary: str) -> LLMAssistantTurn:
        self.tool_trace.append("finish_task")
        self._emit("task_finishing", {"summary": summary})
        return LLMAssistantTurn(
            None,
            (LLMToolCall(
                f"demo-call-{len(self.tool_trace)}",
                "finish_task",
                json.dumps({
                    "status": "completed",
                    "summary": summary,
                    "open_questions": [],
                }),
            ),),
            "tool_calls",
        )

    def _emit_new_observations(
        self,
        payloads: list[Mapping[str, Any]],
        task_id: str,
    ) -> None:
        seen = self._seen_tool_payloads.get(task_id, 0)
        for payload in payloads[seen:]:
            self._emit(
                "tool_succeeded" if payload.get("ok") else "tool_failed",
                {
                    "task_id": task_id,
                    "tool_name": payload.get("tool_name"),
                    "error": payload.get("error"),
                },
            )
        self._seen_tool_payloads[task_id] = len(payloads)

    def _emit(self, event_type: str, data: Mapping[str, Any]) -> None:
        if self.on_event is not None:
            self.on_event(event_type, data)
        if self.step_delay:
            time.sleep(self.step_delay)


def run_demo(
    repository_root: str | Path,
    *,
    project_id: str | None = None,
    workspace: str | Path | None = None,
    simulate_failure: bool = True,
    on_event: DemoEventSink | None = None,
    step_delay: float = 0,
) -> DemoResult:
    """Run a deterministic end-to-end demo in an isolated workspace."""

    root = Path(repository_root).resolve()
    source_context = (
        ResearchContext.load(root, project_id)
        if project_id is not None
        else ResearchContext.load_active(root)
    )
    demo_root = (
        Path(workspace).resolve()
        if workspace is not None
        else Path(tempfile.mkdtemp(prefix="research-agent-demo-"))
    )
    if demo_root.exists() and any(demo_root.iterdir()):
        raise DemoError(f"Demo workspace must be empty: {demo_root}")
    demo_root.mkdir(parents=True, exist_ok=True)

    context, paper_id, evidence_id, evidence_type = _prepare_workspace(
        source_context, demo_root
    )
    _emit(on_event, "workspace_prepared", {
        "workspace": str(demo_root),
        "project_id": context.project_id,
        "source_project_id": source_context.project_id,
        "paper_id": paper_id,
    })
    toolkit = ResearchToolkit(context)
    dispatcher = ResearchToolDispatcher(toolkit)
    provider = DemoProvider(
        paper_id,
        evidence_id,
        fail_once=simulate_failure,
        on_event=on_event,
        step_delay=step_delay,
    )
    executor = ResearchTaskExecutor(
        context,
        provider,
        dispatcher,
        state_store=RunStateStore(context),
        max_rounds=8,
    )
    plan = _demo_plan(context, paper_id, evidence_type)
    validate_agent_plan(plan, context).raise_for_errors()
    _emit(on_event, "plan_validated", {
        "plan_id": plan.plan_id,
        "objective": plan.objective,
        "tasks": [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "question": task.question,
                "dependencies": list(task.dependencies),
            }
            for task in plan.tasks
        ],
    })
    plan_store = ResearchPlanStore(context)
    plan_store.save(plan)
    runner = ResearchPlanExecutor(context, executor)

    first = runner.execute(plan)
    if simulate_failure:
        if first.status != "blocked":
            raise DemoError("Fault-injection run did not stop as blocked")
        _emit(on_event, "run_blocked", {
            "task_id": first.stopped_task_id,
            "message": first.message,
        })
        _emit(on_event, "recovery_started", {
            "action": "retry_blocked",
            "task_id": first.stopped_task_id,
        })
        resumed = runner.execute(plan, retry_blocked=True)
    else:
        resumed = first
    if resumed.status != "completed":
        raise DemoError(f"Demo did not complete after recovery: {resumed.status}")

    synthesis_state = executor.state_store.load("demo-synthesize")
    if synthesis_state is None or not synthesis_state.artifact_ids:
        raise DemoError("Demo synthesis did not record an artifact")
    output = toolkit.read_output(synthesis_state.artifact_ids[0])
    artifact_path = _absolute_artifact_path(context, output.artifact.path)
    result = DemoResult(
        workspace=demo_root,
        plan_path=plan_store.path,
        plan_id=plan.plan_id,
        planned_task_ids=tuple(task.task_id for task in plan.tasks),
        first_attempt_status=first.status,
        first_attempt_stopped_task=first.stopped_task_id,
        resumed_status=resumed.status,
        completed_task_ids=resumed.completed_task_ids,
        unsafe_citation_keys=provider.unsafe_citation_keys,
        artifact_path=artifact_path,
        manifest_path=artifact_path.with_name(f"{artifact_path.name}.manifest.json"),
        tool_trace=tuple(provider.tool_trace),
    )
    _emit(on_event, "run_completed", {
        "status": result.resumed_status,
        "completed_task_ids": list(result.completed_task_ids),
        "artifact_path": str(result.artifact_path),
        "citation_guard_passed": True,
    })
    return result


def _prepare_workspace(
    source: ResearchContext,
    demo_root: Path,
) -> tuple[ResearchContext, str, str, str]:
    source_tools = ResearchToolkit(source)
    selected = None
    for paper in source_tools.list_papers():
        note = source_tools.load_note(paper.paper_id)
        evidence = source_tools.list_evidence(paper_id=paper.paper_id)
        bibliography = source_tools.bibliography_store.for_paper(paper.paper_id)
        extraction = source.extraction_directory / (
            f"{paper.paper_id.replace(':', '_')}.pages.jsonl"
        )
        if note is not None and evidence and bibliography is not None and extraction.is_file():
            selected = (paper, note, evidence, bibliography, extraction)
            break
    if selected is None:
        raise DemoError(
            "The project needs one paper with extraction, note, evidence, and bibliography"
        )

    paper, note, evidence, bibliography, extraction = selected
    project_dir = demo_root / "projects" / "demo"
    output_dir = project_dir / "outputs"
    project_dir.mkdir(parents=True)
    config = {
        "title": "Offline Research Agent Demo",
        "research_question": "How can an agent produce evidence-grounded research writing?",
        "description": "An isolated interview demo using one structured paper.",
        "uploaded_papers": [],
        "seed_papers": [paper.title],
        "research_scope": ["Evidence-backed analysis and citation safety"],
        "output_directory": str(output_dir),
    }
    _write_json(project_dir / "project.json", config)
    _write_jsonl(project_dir / "catalog.jsonl", (paper,))
    _write_jsonl(project_dir / "bibliography.jsonl", (bibliography,))
    _write_jsonl(project_dir / "evidence.jsonl", evidence)
    notes_dir = project_dir / "notes"
    notes_dir.mkdir()
    _write_jsonl(
        notes_dir / f"{paper.paper_id.replace(':', '_')}.json",
        (note,),
    )
    extraction_dir = project_dir / "extracted"
    extraction_dir.mkdir()
    shutil.copyfile(extraction, extraction_dir / extraction.name)
    context = ResearchContext.load(demo_root, "demo")
    return context, paper.paper_id, evidence[0].evidence_id, evidence[0].evidence_type


def _demo_plan(
    context: ResearchContext,
    paper_id: str,
    evidence_type: str,
) -> ResearchPlan:
    return ResearchPlan(
        plan_id="interview-demo-plan",
        project_id=context.project_id,
        objective=context.project.research_question,
        subquestions=("Can the selected claim be traced to a paper page?",),
        tasks=(
            ResearchTask(
                task_id="demo-analyze",
                question="Inspect the selected paper, note, and evidence.",
                task_type="paper_analysis",
                retrieval_queries=("motion tracking method limitations",),
                paper_ids=(paper_id,),
                required_evidence_types=(evidence_type,),
            ),
            ResearchTask(
                task_id="demo-synthesize",
                question="Write a short evidence-backed synthesis.",
                task_type="synthesis",
                dependencies=("demo-analyze",),
            ),
            ResearchTask(
                task_id="demo-audit",
                question="Reject an unsafe candidate and audit the final artifact.",
                task_type="citation_audit",
                dependencies=("demo-synthesize",),
            ),
        ),
        deliverables=("A citation-safe synthesis with provenance",),
        completion_criteria=(
            "The final artifact resolves citations and records provenance.",
        ),
    )


def _task_prompt(messages: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    for message in messages:
        content = message.get("content")
        if message.get("role") != "user" or not isinstance(content, str):
            continue
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "task" in value:
            return value
    raise DemoError("Executor messages do not contain a task prompt")


def _tool_payloads(messages: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    payloads: list[Mapping[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") != "tool" or not isinstance(content, str):
            continue
        value = json.loads(content)
        if isinstance(value, dict):
            payloads.append(value)
    return payloads


def _dependency_artifact_id(task_prompt: Mapping[str, Any]) -> str:
    for state in task_prompt.get("dependency_results", []):
        artifact_ids = state.get("artifact_ids", [])
        if artifact_ids:
            return artifact_ids[0]
    raise DemoError("Audit task did not receive a synthesis artifact")


def _absolute_artifact_path(context: ResearchContext, path: Path) -> Path:
    return path if path.is_absolute() else context.repository_root / path


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, records: Sequence[Any]) -> None:
    path.write_text(
        "".join(
            json.dumps(record_to_dict(record), ensure_ascii=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _emit(
    sink: DemoEventSink | None,
    event_type: str,
    data: Mapping[str, Any],
) -> None:
    if sink is not None:
        sink(event_type, data)
