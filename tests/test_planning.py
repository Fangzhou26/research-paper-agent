from dataclasses import replace
from pathlib import Path

import pytest

from research_agent import (
    PlanConflictError,
    PlanValidationError,
    ResearchContext,
    ResearchPlan,
    ResearchPlanStore,
    ResearchTask,
    list_papers,
    ready_tasks,
    schedule_tasks,
    validate_plan,
    validate_agent_plan,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def valid_plan() -> ResearchPlan:
    paper_ids = tuple(paper.paper_id for paper in list_papers(context()))
    return ResearchPlan(
        plan_id="plan-1",
        project_id=context().project_id,
        objective="Produce an evidence-backed synthesis of uploaded papers.",
        assumptions=("The current corpus defines the search boundary.",),
        subquestions=(
            "What tracking methods are represented?",
            "How do the methods differ?",
        ),
        tasks=(
            ResearchTask(
                task_id="overview",
                question="Which papers and themes are in the corpus?",
                task_type="corpus_overview",
            ),
            ResearchTask(
                task_id="analyze",
                question="What does the selected paper propose?",
                task_type="paper_analysis",
                retrieval_queries=("method contribution limitation",),
                paper_ids=paper_ids,
                required_evidence_types=("method", "limitation"),
                dependencies=("overview",),
            ),
            ResearchTask(
                task_id="synthesize",
                question="How should the evidence be synthesized?",
                task_type="synthesis",
                dependencies=("analyze",),
            ),
            ResearchTask(
                task_id="audit",
                question="Are all citation keys valid?",
                task_type="citation_audit",
                dependencies=("synthesize",),
            ),
        ),
        deliverables=("A cited research synthesis",),
        completion_criteria=(
            "Required evidence is recorded",
            "Citation audit has no unknown keys",
        ),
    )


def issue_codes(plan: ResearchPlan) -> set[str]:
    return {issue.code for issue in validate_plan(plan, context()).issues}


def test_valid_plan_schedules_in_dependency_order() -> None:
    plan = valid_plan()

    assert validate_plan(plan, context()).valid is True
    assert [task.task_id for task in schedule_tasks(plan, context())] == [
        "overview",
        "analyze",
        "synthesize",
        "audit",
    ]
    assert [task.task_id for task in ready_tasks(plan, context(), ())] == ["overview"]
    assert [task.task_id for task in ready_tasks(plan, context(), ("overview",))] == [
        "analyze"
    ]


def test_validator_rejects_unsupported_capability() -> None:
    plan = valid_plan()
    unsupported = replace(
        plan.tasks[1],
        task_type="literature_search",
        retrieval_queries=("external query",),
    )

    assert "unsupported_task_type" in issue_codes(
        replace(plan, tasks=(plan.tasks[0], unsupported, *plan.tasks[2:]))
    )


def test_validator_rejects_missing_analysis_query_and_unknown_paper() -> None:
    plan = valid_plan()
    invalid = replace(
        plan.tasks[1],
        retrieval_queries=(),
        paper_ids=("unknown-paper",),
    )
    codes = issue_codes(replace(plan, tasks=(plan.tasks[0], invalid, *plan.tasks[2:])))

    assert {"missing_retrieval_queries", "unknown_paper_id"}.issubset(codes)


def test_validator_rejects_dependency_cycle() -> None:
    plan = valid_plan()
    first = replace(plan.tasks[0], dependencies=("analyze",))
    cyclic = replace(plan, tasks=(first, *plan.tasks[1:]))

    assert "dependency_cycle" in issue_codes(cyclic)
    with pytest.raises(PlanValidationError):
        schedule_tasks(cyclic, context())


def test_validator_rejects_wrong_project_and_missing_plan_outputs() -> None:
    plan = replace(
        valid_plan(),
        project_id="another-project",
        subquestions=(),
        deliverables=(),
        completion_criteria=(),
    )
    codes = issue_codes(plan)

    assert {
        "project_mismatch",
        "missing_subquestions",
        "missing_deliverables",
        "missing_completion_criteria",
    }.issubset(codes)


def test_plan_store_round_trip_and_explicit_update(tmp_path: Path) -> None:
    store = ResearchPlanStore(context(), tmp_path / "research_plan.json")
    original = valid_plan()
    updated = replace(original, objective="Updated research objective.")

    assert store.save(original).created is True
    assert store.save(original).created is False
    assert store.load() == original
    with pytest.raises(PlanConflictError, match="already exists"):
        store.save(updated)
    result = store.save(updated, overwrite=True)

    assert result.updated is True
    assert store.load() == updated


def test_ready_tasks_rejects_unknown_completed_task() -> None:
    with pytest.raises(PlanValidationError, match="Unknown completed task"):
        ready_tasks(valid_plan(), context(), ("not-in-plan",))


def test_mvp_agent_validator_accepts_minimal_progression() -> None:
    assert validate_agent_plan(valid_plan(), context()).valid is True


def test_mvp_agent_validator_rejects_extra_stage_and_deliverables() -> None:
    plan = valid_plan()
    taxonomy = ResearchTask(
        task_id="taxonomy",
        question="Create a taxonomy.",
        task_type="taxonomy",
        required_evidence_types=("method",),
        dependencies=("analyze",),
    )
    invalid = replace(
        plan,
        tasks=(*plan.tasks[:2], taxonomy, *plan.tasks[2:]),
        deliverables=("Draft", "Matrix"),
    )

    codes = {issue.code for issue in validate_agent_plan(invalid, context()).issues}

    assert "unsupported_task_type" in codes
    assert "invalid_deliverable_count" in codes
    assert "non_analysis_evidence_requirement" in codes
