"""LLM-backed decomposition of a project objective into a validated plan."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import Any, Mapping

from .catalog import PaperCatalog
from .llm import LLMConfig, LLMProvider, LLMProviderError
from .models import ResearchPlan, SchemaValidationError, record_from_dict
from .planning import MVP_AGENT_TASK_TYPES, PlanValidationError, validate_agent_plan
from .project import ResearchContext


PLANNER_SYSTEM_PROMPT = """You are a research planner for a deliberately small MVP agent. Create the shortest executable plan that can produce one evidence-backed final output.
Return only one JSON object matching the supplied contract. Use at most five tasks and only the listed task types and paper IDs. Prefer this progression: one corpus_overview, one or two paper_analysis tasks, one synthesis, then one citation_audit that directly depends on synthesis. Assign every catalog paper ID to exactly one paper_analysis task; group papers thematically instead of creating more tasks. Fold taxonomy and comparison reasoning into synthesis; never create separate tasks for them. Define exactly one deliverable. Only paper_analysis tasks may set required_evidence_types; all other tasks must use an empty array. Do not invent papers, citations, facts, or unavailable capabilities. Tasks must form an acyclic graph. Each paper_analysis task must have retrieval queries. Keep every task pending."""


class PlannerError(RuntimeError):
    """Raised when the model cannot produce a valid executable plan."""


class LLMResearchPlanner:
    """Generate, validate, and optionally repair a research plan."""

    def __init__(
        self,
        context: ResearchContext,
        provider: LLMProvider,
        *,
        max_repair_attempts: int = 1,
    ) -> None:
        self.context = context
        self.provider = provider
        self.max_repair_attempts = max_repair_attempts

    @classmethod
    def from_config(
        cls,
        context: ResearchContext,
        config: LLMConfig,
        *,
        provider: LLMProvider | None = None,
    ) -> "LLMResearchPlanner":
        if provider is None:
            from .llm import OpenAICompatibleProvider

            provider = OpenAICompatibleProvider(config)
        return cls(
            context,
            provider,
            max_repair_attempts=config.max_repair_attempts,
        )

    def create_plan(self, objective: str | None = None) -> ResearchPlan:
        requested_objective = objective or self.context.project.research_question
        messages: list[Mapping[str, str]] = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": self._project_prompt(requested_objective)},
        ]
        last_error = "unknown validation failure"
        for attempt in range(self.max_repair_attempts + 1):
            try:
                candidate = self.provider.generate_json(messages)
                plan = _plan_from_mapping(candidate)
                plan = replace(plan, objective=requested_objective)
                validate_agent_plan(plan, self.context).raise_for_errors()
                return plan
            except (LLMProviderError, SchemaValidationError, PlanValidationError) as exc:
                last_error = str(exc)
                if attempt >= self.max_repair_attempts:
                    break
                messages = [
                    *messages,
                    {"role": "assistant", "content": json.dumps(candidate) if 'candidate' in locals() else "Invalid JSON response"},
                    {
                        "role": "user",
                        "content": (
                            "The proposed plan was invalid. Correct it and return the complete JSON object only. "
                            f"Validation errors: {last_error}"
                        ),
                    },
                ]
        raise PlannerError(
            f"Could not produce a valid research plan after "
            f"{self.max_repair_attempts + 1} attempt(s): {last_error}"
        )

    def _project_prompt(self, objective: str) -> str:
        project = self.context.project
        papers = [
            {"paper_id": paper.paper_id, "title": paper.title}
            for paper in PaperCatalog(self.context.catalog_path).all()
        ]
        payload: dict[str, Any] = {
            "project_id": self.context.project_id,
            "title": project.title,
            "objective": objective,
            "description": project.description,
            "research_scope": list(project.research_scope),
            "seed_papers": list(project.seed_papers),
            "catalog_papers": papers,
            "available_task_types": sorted(MVP_AGENT_TASK_TYPES),
            "output_contract": _plan_contract(),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _plan_from_mapping(data: Mapping[str, Any]) -> ResearchPlan:
    serialized = dict(data)
    serialized.setdefault("schema_version", 1)
    serialized.setdefault("record_type", "ResearchPlan")
    record = record_from_dict(serialized)
    if not isinstance(record, ResearchPlan):
        raise SchemaValidationError("Planner output is not a ResearchPlan")
    return record


def _plan_contract() -> Mapping[str, Any]:
    return {
        "schema_version": 1,
        "record_type": "ResearchPlan",
        "plan_id": "non-empty stable identifier",
        "project_id": "must equal the supplied project_id",
        "objective": "non-empty string",
        "assumptions": ["string"],
        "subquestions": ["string"],
        "tasks": [
            {
                "task_id": "unique string",
                "question": "string",
                "task_type": "one available task type",
                "retrieval_queries": ["string"],
                "paper_ids": ["catalog paper_id"],
                "required_evidence_types": ["string"],
                "dependencies": ["earlier task_id"],
                "status": "pending",
            }
        ],
        "deliverables": ["string"],
        "completion_criteria": ["string"],
    }
