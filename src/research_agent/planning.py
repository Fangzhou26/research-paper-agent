"""Validation, scheduling, and storage for decomposed research plans."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from .catalog import PaperCatalog
from .models import ResearchPlan, ResearchTask, SchemaValidationError, record_from_dict, record_to_dict
from .project import ResearchContext


MVP_TASK_TYPES = frozenset(
    {"corpus_overview", "paper_analysis", "comparison", "taxonomy", "synthesis", "citation_audit"}
)
MVP_TASK_STATUSES = frozenset({"pending", "in_progress", "completed", "blocked"})
QUERY_REQUIRED_TASK_TYPES = frozenset({"paper_analysis"})
MVP_AGENT_TASK_TYPES = frozenset(
    {"corpus_overview", "paper_analysis", "synthesis", "citation_audit"}
)


class PlanError(RuntimeError):
    """Base error for invalid or unreadable research plans."""


class PlanValidationError(PlanError):
    """Raised when a plan cannot be executed safely."""

    def __init__(self, issues: tuple["PlanIssue", ...]) -> None:
        self.issues = issues
        super().__init__("; ".join(issue.message for issue in issues))


class PlanConflictError(PlanError):
    """Raised when plan storage would overwrite different content."""


@dataclass(frozen=True)
class PlanIssue:
    code: str
    message: str
    task_id: str | None = None


@dataclass(frozen=True)
class PlanValidationResult:
    issues: tuple[PlanIssue, ...]

    @property
    def valid(self) -> bool:
        return not self.issues

    def raise_for_errors(self) -> None:
        if self.issues:
            raise PlanValidationError(self.issues)


@dataclass(frozen=True)
class PlanWriteResult:
    plan: ResearchPlan
    created: bool
    updated: bool


def validate_plan(
    plan: ResearchPlan,
    context: ResearchContext,
    *,
    available_task_types: Iterable[str] = MVP_TASK_TYPES,
    max_tasks: int = 20,
) -> PlanValidationResult:
    """Validate a candidate plan against project data and available tools."""

    issues: list[PlanIssue] = []
    available = set(available_task_types)
    if plan.project_id != context.project_id:
        issues.append(PlanIssue("project_mismatch", f"Plan project '{plan.project_id}' does not match '{context.project_id}'"))
    if not plan.subquestions:
        issues.append(PlanIssue("missing_subquestions", "Plan must decompose the objective"))
    if not plan.tasks:
        issues.append(PlanIssue("missing_tasks", "Plan must contain at least one task"))
    if len(plan.tasks) > max_tasks:
        issues.append(PlanIssue("too_many_tasks", f"Plan has {len(plan.tasks)} tasks; maximum is {max_tasks}"))
    if not plan.deliverables:
        issues.append(PlanIssue("missing_deliverables", "Plan must define a deliverable"))
    if not plan.completion_criteria:
        issues.append(PlanIssue("missing_completion_criteria", "Plan must define completion criteria"))

    task_ids = [task.task_id for task in plan.tasks]
    duplicate_ids = sorted(task_id for task_id in set(task_ids) if task_ids.count(task_id) > 1)
    for task_id in duplicate_ids:
        issues.append(PlanIssue("duplicate_task_id", f"Task ID is duplicated: {task_id}", task_id))
    known_task_ids = set(task_ids)
    known_paper_ids = {record.paper_id for record in PaperCatalog(context.catalog_path).all()}
    for task in plan.tasks:
        if task.task_type not in available:
            issues.append(PlanIssue("unsupported_task_type", f"Task type is unavailable: {task.task_type}", task.task_id))
        if task.status not in MVP_TASK_STATUSES:
            issues.append(PlanIssue("unsupported_task_status", f"Task status is invalid: {task.status}", task.task_id))
        if task.task_type in QUERY_REQUIRED_TASK_TYPES and not task.retrieval_queries:
            issues.append(PlanIssue("missing_retrieval_queries", "Paper-analysis task requires at least one retrieval query", task.task_id))
        unknown_papers = sorted(set(task.paper_ids) - known_paper_ids)
        if unknown_papers:
            issues.append(PlanIssue("unknown_paper_id", f"Task references unknown papers: {', '.join(unknown_papers)}", task.task_id))
        if task.task_id in task.dependencies:
            issues.append(PlanIssue("self_dependency", "Task cannot depend on itself", task.task_id))
        unknown_dependencies = sorted(set(task.dependencies) - known_task_ids)
        if unknown_dependencies:
            issues.append(PlanIssue("unknown_dependency", f"Task has unknown dependencies: {', '.join(unknown_dependencies)}", task.task_id))

    dependency_errors = {"self_dependency", "unknown_dependency"}
    if not duplicate_ids and not any(issue.code in dependency_errors for issue in issues):
        if _has_dependency_cycle(plan.tasks):
            issues.append(PlanIssue("dependency_cycle", "Plan task dependencies contain a cycle"))
    return PlanValidationResult(tuple(issues))


def validate_agent_plan(
    plan: ResearchPlan,
    context: ResearchContext,
) -> PlanValidationResult:
    """Validate the deliberately small plan shape used by the first agent MVP."""

    issues = list(validate_plan(
        plan,
        context,
        available_task_types=MVP_AGENT_TASK_TYPES,
        max_tasks=5,
    ).issues)
    if len(plan.deliverables) != 1:
        issues.append(PlanIssue(
            "invalid_deliverable_count",
            "MVP agent plan must define exactly one final deliverable",
        ))
    synthesis = [task for task in plan.tasks if task.task_type == "synthesis"]
    audits = [task for task in plan.tasks if task.task_type == "citation_audit"]
    if len(synthesis) != 1:
        issues.append(PlanIssue(
            "invalid_synthesis_count",
            "MVP agent plan must contain exactly one synthesis task",
        ))
    if len(audits) != 1:
        issues.append(PlanIssue(
            "invalid_citation_audit_count",
            "MVP agent plan must contain exactly one citation-audit task",
        ))
    if len(synthesis) == 1 and len(audits) == 1:
        if synthesis[0].task_id not in audits[0].dependencies:
            issues.append(PlanIssue(
                "audit_before_synthesis",
                "Citation-audit task must depend directly on the synthesis task",
                audits[0].task_id,
            ))
    for task in plan.tasks:
        if task.task_type != "paper_analysis" and task.required_evidence_types:
            issues.append(PlanIssue(
                "non_analysis_evidence_requirement",
                "Only paper-analysis tasks may define required evidence types",
                task.task_id,
            ))
    catalog_papers = {record.paper_id for record in PaperCatalog(context.catalog_path).all()}
    assigned_papers = {
        paper_id
        for task in plan.tasks
        if task.task_type == "paper_analysis"
        for paper_id in task.paper_ids
    }
    unassigned = sorted(catalog_papers - assigned_papers)
    if unassigned:
        issues.append(PlanIssue(
            "unassigned_catalog_papers",
            "MVP agent plan must assign every catalog paper to paper analysis: "
            + ", ".join(unassigned),
        ))
    return PlanValidationResult(tuple(issues))


def schedule_tasks(plan: ResearchPlan, context: ResearchContext) -> tuple[ResearchTask, ...]:
    """Return a stable topological task order after full validation."""

    validate_plan(plan, context).raise_for_errors()
    remaining = list(plan.tasks)
    completed: set[str] = set()
    ordered: list[ResearchTask] = []
    while remaining:
        ready = [task for task in remaining if set(task.dependencies).issubset(completed)]
        if not ready:
            raise PlanValidationError((PlanIssue("dependency_cycle", "No task can be scheduled"),))
        for task in ready:
            ordered.append(task)
            completed.add(task.task_id)
            remaining.remove(task)
    return tuple(ordered)


def ready_tasks(
    plan: ResearchPlan,
    context: ResearchContext,
    completed_task_ids: Iterable[str],
) -> tuple[ResearchTask, ...]:
    """Return pending tasks whose dependencies have completed."""

    validate_plan(plan, context).raise_for_errors()
    completed = set(completed_task_ids)
    known = {task.task_id for task in plan.tasks}
    unknown = completed - known
    if unknown:
        raise PlanValidationError((PlanIssue("unknown_completed_task", f"Unknown completed task IDs: {', '.join(sorted(unknown))}"),))
    return tuple(
        task for task in plan.tasks
        if task.task_id not in completed
        and task.status == "pending"
        and set(task.dependencies).issubset(completed)
    )


class ResearchPlanStore:
    """Persist one current plan for a project with explicit updates."""

    def __init__(self, context: ResearchContext, path: str | Path | None = None) -> None:
        self.context = context
        self.path = Path(path) if path is not None else context.research_plan_path

    def load(self) -> ResearchPlan | None:
        if not self.path.exists():
            return None
        try:
            record = record_from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
            raise PlanError(f"Cannot read research plan: {self.path}") from exc
        if not isinstance(record, ResearchPlan):
            raise PlanError(f"Stored record is not a ResearchPlan: {self.path}")
        return record

    def save(self, plan: ResearchPlan, *, overwrite: bool = False) -> PlanWriteResult:
        validate_plan(plan, self.context).raise_for_errors()
        existing = self.load()
        if existing is not None:
            if existing == plan:
                return PlanWriteResult(plan=existing, created=False, updated=False)
            if not overwrite:
                raise PlanConflictError(f"Research plan already exists with different content: {self.path}")
        content = json.dumps(record_to_dict(plan), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        _write_text_atomic(self.path, content)
        return PlanWriteResult(plan=plan, created=existing is None, updated=existing is not None)


def _has_dependency_cycle(tasks: tuple[ResearchTask, ...]) -> bool:
    dependencies = {task.task_id: set(task.dependencies) for task in tasks}
    remaining = set(dependencies)
    while remaining:
        ready = {task_id for task_id in remaining if not (dependencies[task_id] & remaining)}
        if not ready:
            return True
        remaining -= ready
    return False


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise PlanError(f"Cannot write research plan: {path}") from exc
