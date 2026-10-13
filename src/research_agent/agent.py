"""Unified entry point for the minimal reusable research agent."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionEventSink, ResearchTaskExecutor
from .llm import LLMConfig, OpenAICompatibleProvider
from .llm_planner import LLMResearchPlanner
from .models import ResearchPlan
from .plan_execution import PlanExecutionResult, ResearchPlanExecutor
from .planning import PlanWriteResult, ResearchPlanStore
from .project import ResearchContext
from .tool_dispatch import ResearchToolDispatcher
from .toolkit import ResearchToolkit


class ResearchAgentError(RuntimeError):
    """Raised when one agent run cannot be started safely."""


@dataclass(frozen=True)
class AgentRunResult:
    """Plan and execution outcome from one agent invocation."""

    plan: ResearchPlan
    plan_created: bool
    execution: PlanExecutionResult


class ResearchAgent:
    """Coordinate planning, persistence, and dependency-aware execution."""

    def __init__(
        self,
        context: ResearchContext,
        planner: LLMResearchPlanner,
        plan_store: ResearchPlanStore,
        plan_executor: ResearchPlanExecutor,
    ) -> None:
        self.context = context
        self.planner = planner
        self.plan_store = plan_store
        self.plan_executor = plan_executor

    @classmethod
    def from_config(
        cls,
        context: ResearchContext,
        config: LLMConfig,
        *,
        max_task_rounds: int = 12,
        on_event: ExecutionEventSink | None = None,
    ) -> "ResearchAgent":
        provider = OpenAICompatibleProvider(config)
        planner = LLMResearchPlanner.from_config(context, config, provider=provider)
        toolkit = ResearchToolkit(context)
        dispatcher = ResearchToolDispatcher(toolkit)
        task_executor = ResearchTaskExecutor(
            context,
            provider,
            dispatcher,
            max_rounds=max_task_rounds,
            on_event=on_event,
        )
        return cls(
            context,
            planner,
            ResearchPlanStore(context),
            ResearchPlanExecutor(context, task_executor),
        )

    def run(
        self,
        objective: str | None = None,
        *,
        resume: bool = True,
        retry_blocked: bool = False,
    ) -> AgentRunResult:
        prepared = self.prepare_plan(objective)
        execution = self.plan_executor.execute(
            prepared.plan,
            resume=resume,
            retry_blocked=retry_blocked,
        )
        return AgentRunResult(
            plan=prepared.plan,
            plan_created=prepared.created,
            execution=execution,
        )

    def prepare_plan(self, objective: str | None = None) -> PlanWriteResult:
        """Load the stored plan or generate and persist one without executing it."""

        requested = objective.strip() if isinstance(objective, str) else objective
        if objective is not None and not requested:
            raise ResearchAgentError("Research objective must not be empty")
        plan = self.plan_store.load()
        if plan is not None:
            if requested is not None and plan.objective != requested:
                raise ResearchAgentError(
                    "A stored research plan has a different objective; explicit replanning is required"
                )
            return PlanWriteResult(plan=plan, created=False, updated=False)
        plan = self.planner.create_plan(requested)
        return self.plan_store.save(plan)

    def replace_unstarted_plan(self, objective: str | None = None) -> PlanWriteResult:
        """Replace a stored plan only when none of its tasks has a checkpoint."""

        existing = self.plan_store.load()
        if existing is None:
            return self.prepare_plan(objective)
        state_store = self.plan_executor.task_executor.state_store
        started = [
            task.task_id for task in existing.tasks
            if state_store.load(task.task_id) is not None
        ]
        if started:
            raise ResearchAgentError(
                "Cannot replace a started plan; task checkpoints exist: "
                + ", ".join(started)
            )
        requested = objective.strip() if isinstance(objective, str) else objective
        if objective is not None and not requested:
            raise ResearchAgentError("Research objective must not be empty")
        replacement = self.planner.create_plan(requested)
        return self.plan_store.save(replacement, overwrite=True)
