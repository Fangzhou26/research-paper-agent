import json
from pathlib import Path

import pytest

from research_agent import (
    ArtifactRecord,
    BibliographyEntry,
    EvidenceRecord,
    PageText,
    PaperNote,
    PaperRecord,
    ResearchPlan,
    ResearchTask,
    RunState,
    SCHEMA_VERSION,
    SchemaValidationError,
    SearchCandidate,
    record_from_dict,
    record_to_dict,
)


RECORDS = [
    PaperRecord(
        paper_id="sha256:abc",
        title="A Paper",
        authors=("Ada Author", "Rui Researcher"),
        year=2026,
        doi="10.0000/example",
        source_type="uploaded",
        source_uri="upload://paper.pdf",
        file_path=Path("papers/sha256-abc/source.pdf"),
        verification_status="source_verified",
        tags=("topic-a",),
    ),
    PageText(
        paper_id="sha256:abc",
        page=3,
        text="Extracted text.",
        extraction_method="native",
        quality="good",
    ),
    EvidenceRecord(
        evidence_id="evidence-1",
        project_id="example",
        paper_id="sha256:abc",
        page=3,
        passage="Reported result.",
        interpretation="The method improves the reported metric.",
        evidence_type="result",
        support_type="direct",
    ),
    PaperNote(
        project_id="example",
        paper_id="sha256:abc",
        research_problem="A research problem.",
        method="A research method.",
        data_or_inputs=("Dataset A",),
        key_contributions=("Contribution A",),
        main_results=("Result A",),
        limitations=("Limitation A",),
        relevance="Relevant to the active question.",
        topics=("topic-a",),
        evidence_ids=("evidence-1",),
        uncertainties=("Venue not confirmed",),
    ),
    SearchCandidate(
        candidate_id="candidate-1",
        title="Candidate Paper",
        authors=("Casey Candidate",),
        year=2025,
        source_url="https://example.test/paper",
        external_id="doi:10.0000/candidate",
        screening_decision="include",
        screening_reason="Directly studies the research question.",
    ),
    BibliographyEntry(
        citation_key="author2026paper",
        paper_id="sha256:abc",
        entry_type="article",
        fields=(("title", "A Paper"), ("year", "2026")),
        verification_status="source_verified",
    ),
    RunState(
        task_id="task-1",
        project_id="example",
        task_type="related-work",
        status="in_progress",
        current_stage="paper_analysis",
        completed_steps=("intake",),
        artifact_ids=("artifact-1",),
        open_questions=("Confirm inclusion criteria",),
        summary="Initial task checkpoint.",
        updated_at="2026-09-01T12:00:00Z",
    ),
    ArtifactRecord(
        artifact_id="artifact-1",
        project_id="example",
        artifact_type="related_work",
        path=Path("projects/example/outputs/related_work.tex"),
        paper_ids=("sha256:abc",),
        evidence_ids=("evidence-1",),
        created_at="2026-09-01T12:00:00Z",
    ),
    ResearchTask(
        task_id="analyze-1",
        question="What method does the paper propose?",
        task_type="paper_analysis",
        retrieval_queries=("proposed method",),
        paper_ids=("sha256:abc",),
        required_evidence_types=("method",),
        dependencies=("overview",),
    ),
    ResearchPlan(
        plan_id="plan-1",
        project_id="example",
        objective="Synthesize the uploaded literature.",
        assumptions=("Only uploaded papers are available.",),
        subquestions=("Which methods are represented?",),
        tasks=(
            ResearchTask(
                task_id="overview",
                question="What papers are available?",
                task_type="corpus_overview",
            ),
            ResearchTask(
                task_id="synthesis",
                question="How do the methods relate?",
                task_type="synthesis",
                dependencies=("overview",),
            ),
        ),
        deliverables=("A concise synthesis",),
        completion_criteria=("All claims use cataloged papers",),
    ),
]


@pytest.mark.parametrize("record", RECORDS, ids=lambda record: type(record).__name__)
def test_records_round_trip_through_json(record: object) -> None:
    serialized = record_to_dict(record)  # type: ignore[arg-type]
    json_value = json.loads(json.dumps(serialized))

    assert json_value["schema_version"] == SCHEMA_VERSION
    assert record_from_dict(json_value) == record


def test_record_loader_rejects_unknown_schema_version() -> None:
    data = record_to_dict(RECORDS[0])
    data["schema_version"] = SCHEMA_VERSION + 1

    with pytest.raises(SchemaValidationError, match="Unsupported schema version"):
        record_from_dict(data)


def test_record_loader_rejects_unknown_fields() -> None:
    data = record_to_dict(RECORDS[0])
    data["topic_specific_field"] = "must live outside the generic schema"

    with pytest.raises(SchemaValidationError, match="topic_specific_field"):
        record_from_dict(data)


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: PaperRecord(paper_id="", title="Paper"), "paper_id"),
        (
            lambda: PageText(
                paper_id="paper-1", page=0, text="", extraction_method="native"
            ),
            "page",
        ),
        (
            lambda: EvidenceRecord(
                evidence_id="evidence-1",
                project_id="project",
                paper_id="paper-1",
                page=1,
                passage="",
                interpretation="Interpretation",
                evidence_type="method",
            ),
            "passage",
        ),
    ],
)
def test_records_validate_required_values(constructor, message: str) -> None:
    with pytest.raises(SchemaValidationError, match=message):
        constructor()
