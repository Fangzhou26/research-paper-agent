"""Minimal dependency-aware runner for a validated research plan."""

from __future__ import annotations

from dataclasses import dataclass

from .execution import ExecutionError, ResearchTaskExecutor, TaskExecutionResult
from .models import ResearchPlan
from .planning import schedule_tasks, validate_agent_plan
from .project import ResearchContext


@dataclass(frozen=True)
class PlanExecutionResult:
    """Outcome of executing as much of one plan as currently possible."""

    status: str
    completed_task_ids: tuple[str, ...]
    pending_task_ids: tuple[str, ...]
    task_results: tuple[TaskExecutionResult, ...]
    stopped_task_id: str | None = None
    message: str | None = None


class ResearchPlanExecutor:
    """Execute validated plan tasks sequentially in dependency order."""

    def __init__(
        self,
        context: ResearchContext,
        task_executor: ResearchTaskExecutor,
    ) -> None:
        self.context = context
        self.task_executor = task_executor

    def execute(
        self,
        plan: ResearchPlan,
        *,
        resume: bool = True,
        retry_blocked: bool = False,
    ) -> PlanExecutionResult:
        validate_agent_plan(plan, self.context).raise_for_errors()
        ordered = schedule_tasks(plan, self.context)
        completed = {task.task_id for task in ordered if task.status == "completed"}
        results: list[TaskExecutionResult] = []

        for task in ordered:
            if task.task_id in completed:
                continue
            checkpoint = self.task_executor.state_store.load(task.task_id) if resume else None
            if checkpoint is not None and checkpoint.status == "completed":
                completed.add(task.task_id)
                continue
            if (
                checkpoint is not None
                and checkpoint.status in {"blocked", "needs_input"}
                and not (retry_blocked and checkpoint.status == "blocked")
            ):
                return self._result(
                    plan, checkpoint.status, completed, results,
                    stopped_task_id=task.task_id,
                    message=checkpoint.summary,
                )
            if task.status == "blocked":
                return self._result(
                    plan, "blocked", completed, results,
                    stopped_task_id=task.task_id,
                    message="The research plan marks this task as blocked.",
                )
            missing_dependencies = set(task.dependencies) - completed
            if missing_dependencies:
                return self._result(
                    plan, "blocked", completed, results,
                    stopped_task_id=task.task_id,
                    message=(
                        "Task dependencies are not completed: "
                        + ", ".join(sorted(missing_dependencies))
                    ),
                )
            try:
                result = self.task_executor.execute(
                    task,
                    completion_criteria=plan.completion_criteria,
                    dependency_states=tuple(
                        state
                        for dependency_id in task.dependencies
                        if (state := self.task_executor.state_store.load(dependency_id)) is not None
                    ),
                    resume=resume,
                    retry_blocked=retry_blocked,
                )
            except ExecutionError as exc:
                return self._result(
                    plan, "blocked", completed, results,
                    stopped_task_id=task.task_id,
                    message=str(exc),
                )
            results.append(result)
            if result.state.status != "completed":
                return self._result(
                    plan, result.state.status, completed, results,
                    stopped_task_id=task.task_id,
                    message=result.summary,
                )
            completed.add(task.task_id)

        return self._result(
            plan, "completed", completed, results,
            message="All research plan tasks completed.",
        )

    @staticmethod
    def _result(
        plan: ResearchPlan,
        status: str,
        completed: set[str],
        results: list[TaskExecutionResult],
        *,
        stopped_task_id: str | None = None,
        message: str | None = None,
    ) -> PlanExecutionResult:
        return PlanExecutionResult(
            status=status,
            completed_task_ids=tuple(
                task.task_id for task in plan.tasks if task.task_id in completed
            ),
            pending_task_ids=tuple(
                task.task_id for task in plan.tasks if task.task_id not in completed
            ),
            task_results=tuple(results),
            stopped_task_id=stopped_task_id,
            message=message,
        )
