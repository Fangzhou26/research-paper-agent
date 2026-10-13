from pathlib import Path

import pytest

from research_agent import (
    AgentRunResult,
    LLMConfig,
    PlanExecutionResult,
    ResearchAgent,
    ResearchAgentError,
    ResearchContext,
    ResearchPlan,
    ResearchPlanStore,
    ResearchTask,
    RunState,
    RunStateStore,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StubPlanner:
    def __init__(self, generated_plan: ResearchPlan) -> None:
        self.generated_plan = generated_plan
        self.calls: list[str | None] = []

    def create_plan(self, objective=None) -> ResearchPlan:
        self.calls.append(objective)
        return self.generated_plan


class StubPlanExecutor:
    def __init__(self, state_store: RunStateStore) -> None:
        self.calls: list[tuple[ResearchPlan, bool, bool]] = []
        self.task_executor = self
        self.state_store = state_store

    def execute(
        self, plan: ResearchPlan, *, resume: bool = True, retry_blocked: bool = False
    ) -> PlanExecutionResult:
        self.calls.append((plan, resume, retry_blocked))
        return PlanExecutionResult(
            status="completed",
            completed_task_ids=tuple(task.task_id for task in plan.tasks),
            pending_task_ids=(),
            task_results=(),
            message="All tasks completed.",
        )


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def plan(objective: str = "Inspect the corpus.") -> ResearchPlan:
    return ResearchPlan(
        plan_id="agent-plan",
        project_id="demo",
        objective=objective,
        subquestions=("What papers are available?",),
        tasks=(ResearchTask(
            task_id="overview",
            question="What papers are available?",
            task_type="corpus_overview",
        ),),
        deliverables=("Corpus overview",),
        completion_criteria=("Corpus inspected",),
    )


def agent(tmp_path: Path, generated_plan: ResearchPlan):
    planner = StubPlanner(generated_plan)
    plan_executor = StubPlanExecutor(RunStateStore(context(), tmp_path / "runs"))
    research_agent = ResearchAgent(
        context(),
        planner,  # type: ignore[arg-type]
        ResearchPlanStore(context(), tmp_path / "research_plan.json"),
        plan_executor,  # type: ignore[arg-type]
    )
    return research_agent, planner, plan_executor


def test_agent_creates_stores_and_executes_plan(tmp_path: Path) -> None:
    research_agent, planner, plan_executor = agent(tmp_path, plan())

    result = research_agent.run("Inspect the corpus.")

    assert isinstance(result, AgentRunResult)
    assert result.plan_created is True
    assert result.execution.status == "completed"
    assert planner.calls == ["Inspect the corpus."]
    assert research_agent.plan_store.load() == plan()
    assert plan_executor.calls == [(plan(), True, False)]


def test_agent_reuses_stored_plan_without_replanning(tmp_path: Path) -> None:
    research_agent, planner, plan_executor = agent(tmp_path, plan())
    research_agent.plan_store.save(plan())

    result = research_agent.run()

    assert result.plan_created is False
    assert planner.calls == []
    assert plan_executor.calls == [(plan(), True, False)]


def test_agent_rejects_objective_conflict_with_stored_plan(tmp_path: Path) -> None:
    research_agent, planner, plan_executor = agent(tmp_path, plan())
    research_agent.plan_store.save(plan())

    with pytest.raises(ResearchAgentError, match="different objective"):
        research_agent.run("A different objective")

    assert planner.calls == []
    assert plan_executor.calls == []


def test_agent_factory_wires_mvp_without_calling_network() -> None:
    config = LLMConfig.load(REPOSITORY_ROOT / "config/llm.example.json")

    research_agent = ResearchAgent.from_config(context(), config, max_task_rounds=3)

    assert research_agent.context.project_id == "demo"
    assert research_agent.plan_executor.task_executor.max_rounds == 3


def test_agent_replaces_unstarted_plan(tmp_path: Path) -> None:
    replacement = plan("Replacement objective")
    research_agent, planner, _ = agent(tmp_path, replacement)
    research_agent.plan_store.save(plan())

    result = research_agent.replace_unstarted_plan("Replacement objective")

    assert result.updated is True
    assert research_agent.plan_store.load() == replacement
    assert planner.calls == ["Replacement objective"]


def test_agent_refuses_to_replace_plan_with_checkpoint(tmp_path: Path) -> None:
    research_agent, planner, _ = agent(tmp_path, plan("Replacement objective"))
    research_agent.plan_store.save(plan())
    state_store = research_agent.plan_executor.task_executor.state_store
    state_store.save(RunState(
        task_id="overview",
        project_id="demo",
        task_type="corpus_overview",
        status="in_progress",
        current_stage="decide",
    ))

    with pytest.raises(ResearchAgentError, match="checkpoints exist"):
        research_agent.replace_unstarted_plan("Replacement objective")

    assert planner.calls == []
