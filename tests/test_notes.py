from pathlib import Path

import pytest

from research_agent import (
    EvidenceRecord,
    EvidenceStore,
    NoteConflictError,
    NoteStoreError,
    PaperNote,
    PaperNoteStore,
    ResearchContext,
    list_papers,
    read_paper_pages,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def valid_evidence(evidence_id: str = "evidence-1") -> EvidenceRecord:
    project = context()
    paper = list_papers(project)[0]
    page = read_paper_pages(project, paper.paper_id, [1])[0]
    passage = " ".join(page.text.split())[:160]
    return EvidenceRecord(
        evidence_id=evidence_id,
        project_id=project.project_id,
        paper_id=paper.paper_id,
        page=1,
        passage=passage,
        interpretation="This passage identifies the paper and its research context.",
        evidence_type="context",
        support_type="direct",
    )


def test_evidence_store_validates_and_persists_page_passage(tmp_path: Path) -> None:
    store = EvidenceStore(context(), tmp_path / "evidence.jsonl")
    evidence = valid_evidence()

    first = store.add(evidence)
    repeated = store.add(evidence)

    assert first.created is True
    assert repeated.created is False
    assert store.get(evidence.evidence_id) == evidence
    assert store.list(paper_id=evidence.paper_id) == (evidence,)


def test_evidence_store_rejects_passage_not_on_page(tmp_path: Path) -> None:
    store = EvidenceStore(context(), tmp_path / "evidence.jsonl")
    evidence = valid_evidence()
    invalid = EvidenceRecord(
        evidence_id=evidence.evidence_id,
        project_id=evidence.project_id,
        paper_id=evidence.paper_id,
        page=evidence.page,
        passage="This passage definitely does not occur in the source document.",
        interpretation=evidence.interpretation,
        evidence_type=evidence.evidence_type,
    )

    with pytest.raises(NoteStoreError, match="not present on page"):
        store.add(invalid)


def test_note_store_requires_known_evidence(tmp_path: Path) -> None:
    evidence_store = EvidenceStore(context(), tmp_path / "evidence.jsonl")
    note_store = PaperNoteStore(
        context(),
        tmp_path / "notes",
        evidence_store=evidence_store,
    )
    paper = list_papers(context())[0]
    note = PaperNote(
        project_id=context().project_id,
        paper_id=paper.paper_id,
        research_problem="Research problem",
        method="Method",
        evidence_ids=("missing-evidence",),
    )

    with pytest.raises(NoteStoreError, match="Unknown evidence IDs"):
        note_store.save(note)


def test_note_store_supports_explicit_updates(tmp_path: Path) -> None:
    evidence_store = EvidenceStore(context(), tmp_path / "evidence.jsonl")
    evidence = valid_evidence()
    evidence_store.add(evidence)
    note_store = PaperNoteStore(
        context(),
        tmp_path / "notes",
        evidence_store=evidence_store,
    )
    original = PaperNote(
        project_id=context().project_id,
        paper_id=evidence.paper_id,
        research_problem="Research problem",
        method="Original method description",
        evidence_ids=(evidence.evidence_id,),
    )
    updated = PaperNote(
        project_id=context().project_id,
        paper_id=evidence.paper_id,
        research_problem="Research problem",
        method="Updated method description",
        evidence_ids=(evidence.evidence_id,),
    )

    assert note_store.save(original).created is True
    with pytest.raises(NoteConflictError, match="already exists"):
        note_store.save(updated)
    result = note_store.save(updated, overwrite=True)

    assert result.updated is True
    assert note_store.load(evidence.paper_id) == updated
    assert note_store.all() == (updated,)
