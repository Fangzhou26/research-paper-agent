from pathlib import Path
from typing import Mapping, Sequence

import pytest

from research_agent import (
    LLMResearchPlanner,
    PlannerError,
    ResearchContext,
    ResearchPlan,
    list_papers,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StubProvider:
    def __init__(self, responses: list[Mapping[str, object]]) -> None:
        self.responses = responses
        self.messages: list[Sequence[Mapping[str, str]]] = []

    def generate_json(self, messages: Sequence[Mapping[str, str]]) -> Mapping[str, object]:
        self.messages.append(messages)
        return self.responses.pop(0)


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def valid_response() -> dict[str, object]:
    paper_ids = [paper.paper_id for paper in list_papers(context())]
    return {
        "plan_id": "generated-plan",
        "project_id": "demo",
        "objective": "Synthesize the supplied corpus.",
        "assumptions": [],
        "subquestions": ["What methods are represented?"],
        "tasks": [
            {
                "task_id": "overview",
                "question": "What is in the corpus?",
                "task_type": "corpus_overview",
                "retrieval_queries": [],
                "paper_ids": [],
                "required_evidence_types": [],
                "dependencies": [],
                "status": "pending",
            },
            {
                "task_id": "analyze",
                "question": "What method does the paper use?",
                "task_type": "paper_analysis",
                "retrieval_queries": ["method limitations"],
                "paper_ids": paper_ids,
                "required_evidence_types": ["method"],
                "dependencies": ["overview"],
                "status": "pending",
            },
            {
                "task_id": "synthesize",
                "question": "What does the evidence show?",
                "task_type": "synthesis",
                "retrieval_queries": [],
                "paper_ids": [],
                "required_evidence_types": [],
                "dependencies": ["analyze"],
                "status": "pending",
            },
            {
                "task_id": "audit",
                "question": "Are the final citations valid?",
                "task_type": "citation_audit",
                "retrieval_queries": [],
                "paper_ids": [],
                "required_evidence_types": [],
                "dependencies": ["synthesize"],
                "status": "pending",
            },
        ],
        "deliverables": ["Evidence-backed synthesis"],
        "completion_criteria": ["All claims cite recorded evidence"],
    }


def test_planner_builds_and_validates_plan_from_project_context() -> None:
    provider = StubProvider([valid_response()])

    plan = LLMResearchPlanner(context(), provider).create_plan()

    assert isinstance(plan, ResearchPlan)
    assert plan.project_id == "demo"
    assert plan.objective == context().project.research_question
    prompt = provider.messages[0][1]["content"]
    assert context().project.research_question in prompt
    assert list_papers(context())[0].paper_id in prompt


def test_planner_repairs_an_invalid_plan_once() -> None:
    invalid = {**valid_response(), "project_id": "wrong-project"}
    provider = StubProvider([invalid, valid_response()])

    plan = LLMResearchPlanner(context(), provider, max_repair_attempts=1).create_plan()

    assert plan.project_id == "demo"
    assert len(provider.messages) == 2
    assert "project" in provider.messages[1][-1]["content"].lower()


def test_planner_reports_exhausted_validation_attempts() -> None:
    invalid = {**valid_response(), "tasks": []}
    provider = StubProvider([invalid])

    with pytest.raises(PlannerError, match="1 attempt"):
        LLMResearchPlanner(context(), provider, max_repair_attempts=0).create_plan()
