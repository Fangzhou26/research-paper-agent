from dataclasses import replace
from pathlib import Path

from research_agent import (
    BibliographyStore,
    EvidenceStore,
    OutputWriter,
    PaperNoteStore,
    ResearchContext,
    ResearchToolkit,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def toolkit(tmp_path: Path) -> ResearchToolkit:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    context = replace(
        context,
        project=replace(context.project, output_directory=tmp_path / "outputs"),
    )
    evidence_store = EvidenceStore(context, tmp_path / "evidence.jsonl")
    note_store = PaperNoteStore(
        context,
        tmp_path / "notes",
        evidence_store=evidence_store,
    )
    bibliography_store = BibliographyStore(context)
    output_writer = OutputWriter(
        context,
        evidence_store=evidence_store,
        bibliography_store=bibliography_store,
    )
    return ResearchToolkit(
        context,
        evidence_store=evidence_store,
        note_store=note_store,
        bibliography_store=bibliography_store,
        output_writer=output_writer,
    )


def test_minimal_research_toolchain_end_to_end(tmp_path: Path) -> None:
    tools = toolkit(tmp_path)
    match = tools.search_pages("synthetic sample study", limit=1)[0]
    page = tools.read_pages(match.paper_id, [match.page])[0]
    passage = " ".join(page.text.split())[:180]

    evidence = tools.add_evidence(
        match.paper_id,
        match.page,
        passage,
        "The cited page is relevant to the fictional method.",
        "method",
    )
    repeated = tools.add_evidence(
        match.paper_id,
        match.page,
        passage,
        "The cited page is relevant to the fictional method.",
        "method",
    )
    note, write_result = tools.save_note(
        match.paper_id,
        "How can a claim be tied to source evidence?",
        "A fictional method stores page-level provenance.",
        evidence_ids=(evidence.evidence_id,),
    )
    citation_key = tools.citation_key(match.paper_id)
    content = f"The synthetic method stores evidence \\cite{{{citation_key}}}."
    artifact = tools.write_output(
        "agent-smoke-test.tex",
        content,
        paper_ids=(match.paper_id,),
        evidence_ids=(evidence.evidence_id,),
    )

    assert repeated == evidence
    assert write_result.created is True
    assert tools.load_note(match.paper_id) == note
    assert tools.list_evidence(paper_id=match.paper_id) == (evidence,)
    assert tools.audit_citations(content).valid is True
    assert artifact.paper_ids == (match.paper_id,)
    assert (tmp_path / "outputs/agent-smoke-test.tex").is_file()
    assert (tmp_path / "outputs/agent-smoke-test.tex.manifest.json").is_file()
