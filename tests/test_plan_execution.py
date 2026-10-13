from pathlib import Path

from research_agent import (
    PlanExecutionResult,
    ResearchContext,
    ResearchPlan,
    ResearchPlanExecutor,
    ResearchTask,
    RunState,
    RunStateStore,
    TaskExecutionResult,
    list_papers,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StubTaskExecutor:
    def __init__(
        self,
        state_store: RunStateStore,
        outcomes: dict[str, str],
    ) -> None:
        self.state_store = state_store
        self.outcomes = outcomes
        self.calls: list[str] = []
        self.dependency_calls: dict[str, tuple[RunState, ...]] = {}

    def execute(
        self, task, *, completion_criteria=(), dependency_states=(),
        resume=True, retry_blocked=False,
    ):
        self.calls.append(task.task_id)
        self.dependency_calls[task.task_id] = tuple(dependency_states)
        status = self.outcomes.get(task.task_id, "completed")
        state = RunState(
            task_id=task.task_id,
            project_id=self.state_store.context.project_id,
            task_type=task.task_type,
            status=status,
            current_stage="finished",
            artifact_ids=("artifact-synthesis",) if task.task_type == "synthesis" else (),
            summary=f"{task.task_id} {status}",
        )
        self.state_store.save(state)
        return TaskExecutionResult(state, state.summary or "", 1)


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def plan() -> ResearchPlan:
    paper_ids = tuple(paper.paper_id for paper in list_papers(context()))
    return ResearchPlan(
        plan_id="runner-plan",
        project_id="demo",
        objective="Inspect and synthesize the corpus.",
        subquestions=("What is in the corpus?",),
        tasks=(
            ResearchTask(
                task_id="overview",
                question="What papers are available?",
                task_type="corpus_overview",
            ),
            ResearchTask(
                task_id="analysis",
                question="What methods are represented?",
                task_type="paper_analysis",
                retrieval_queries=("methods contributions",),
                paper_ids=paper_ids,
                dependencies=("overview",),
            ),
            ResearchTask(
                task_id="synthesis",
                question="What does the corpus show?",
                task_type="synthesis",
                dependencies=("analysis",),
            ),
            ResearchTask(
                task_id="audit",
                question="Are citation keys valid?",
                task_type="citation_audit",
                dependencies=("synthesis",),
            ),
        ),
        deliverables=("Research synthesis",),
        completion_criteria=("All planned tasks complete",),
    )


def runner(tmp_path: Path, outcomes: dict[str, str]) -> tuple[ResearchPlanExecutor, StubTaskExecutor]:
    store = RunStateStore(context(), tmp_path / "runs")
    task_executor = StubTaskExecutor(store, outcomes)
    return ResearchPlanExecutor(context(), task_executor), task_executor  # type: ignore[arg-type]


def test_plan_runner_executes_tasks_in_dependency_order(tmp_path: Path) -> None:
    plan_runner, task_runner = runner(tmp_path, {})

    result = plan_runner.execute(plan())

    assert isinstance(result, PlanExecutionResult)
    assert result.status == "completed"
    assert result.completed_task_ids == ("overview", "analysis", "synthesis", "audit")
    assert result.pending_task_ids == ()
    assert task_runner.calls == ["overview", "analysis", "synthesis", "audit"]
    assert task_runner.dependency_calls["audit"][0].artifact_ids == ("artifact-synthesis",)


def test_plan_runner_stops_immediately_when_task_needs_input(tmp_path: Path) -> None:
    plan_runner, task_runner = runner(tmp_path, {"synthesis": "needs_input"})

    result = plan_runner.execute(plan())

    assert result.status == "needs_input"
    assert result.completed_task_ids == ("overview", "analysis")
    assert result.pending_task_ids == ("synthesis", "audit")
    assert result.stopped_task_id == "synthesis"
    assert task_runner.calls == ["overview", "analysis", "synthesis"]


def test_plan_runner_resumes_after_completed_checkpoint(tmp_path: Path) -> None:
    plan_runner, task_runner = runner(tmp_path, {})
    task_runner.state_store.save(RunState(
        task_id="overview",
        project_id="demo",
        task_type="corpus_overview",
        status="completed",
        current_stage="finished",
        summary="Previously completed.",
    ))

    result = plan_runner.execute(plan())

    assert result.status == "completed"
    assert task_runner.calls == ["analysis", "synthesis", "audit"]


def test_plan_runner_preserves_blocked_checkpoint_without_model_call(tmp_path: Path) -> None:
    plan_runner, task_runner = runner(tmp_path, {})
    task_runner.state_store.save(RunState(
        task_id="overview",
        project_id="demo",
        task_type="corpus_overview",
        status="blocked",
        current_stage="provider_error",
        summary="Provider unavailable.",
    ))

    result = plan_runner.execute(plan())

    assert result.status == "blocked"
    assert result.stopped_task_id == "overview"
    assert result.message == "Provider unavailable."
    assert task_runner.calls == []


def test_plan_runner_retries_blocked_checkpoint_when_explicit(tmp_path: Path) -> None:
    plan_runner, task_runner = runner(tmp_path, {})
    task_runner.state_store.save(RunState(
        task_id="overview",
        project_id="demo",
        task_type="corpus_overview",
        status="blocked",
        current_stage="finished",
        summary="A tool was unavailable.",
    ))

    result = plan_runner.execute(plan(), retry_blocked=True)

    assert result.status == "completed"
    assert task_runner.calls == ["overview", "analysis", "synthesis", "audit"]
