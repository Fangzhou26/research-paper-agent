from pathlib import Path

import pytest

from research_agent import (
    BibliographyEntry,
    BibliographyError,
    BibliographyStore,
    ResearchContext,
    list_papers,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def entry(key: str = "example2026") -> BibliographyEntry:
    paper = list_papers(context())[0]
    return BibliographyEntry(
        citation_key=key,
        paper_id=paper.paper_id,
        entry_type="misc",
        fields=(("title", paper.title), ("year", str(paper.year))),
        verification_status="metadata_verified",
    )


def test_bibliography_add_lookup_and_idempotency(tmp_path: Path) -> None:
    store = BibliographyStore(context(), tmp_path / "bibliography.jsonl")
    bibliography_entry = entry()

    assert store.add(bibliography_entry).created is True
    assert store.add(bibliography_entry).created is False
    assert store.get(bibliography_entry.citation_key) == bibliography_entry
    assert store.for_paper(bibliography_entry.paper_id) == bibliography_entry
    assert store.citation_key_for(bibliography_entry.paper_id) == "example2026"


def test_bibliography_rejects_unknown_paper(tmp_path: Path) -> None:
    store = BibliographyStore(context(), tmp_path / "bibliography.jsonl")
    invalid = BibliographyEntry(
        citation_key="missing",
        paper_id="missing-paper",
        entry_type="misc",
        fields=(("title", "Missing"),),
    )

    with pytest.raises(BibliographyError, match="Unknown paper ID"):
        store.add(invalid)


def test_bibliography_rejects_key_or_paper_conflict(tmp_path: Path) -> None:
    store = BibliographyStore(context(), tmp_path / "bibliography.jsonl")
    original = entry()
    store.add(original)
    conflicting = BibliographyEntry(
        citation_key="different-key",
        paper_id=original.paper_id,
        entry_type="misc",
        fields=original.fields,
    )

    with pytest.raises(BibliographyError, match="different bibliography metadata"):
        store.add(conflicting)


def test_citation_audit_reports_unknown_and_unused_keys(tmp_path: Path) -> None:
    store = BibliographyStore(context(), tmp_path / "bibliography.jsonl")
    store.add(entry("known"))

    audit = store.audit_text(r"Supported \cite{known,missing} and repeated \citep{known}.")

    assert audit.used_keys == ("known", "missing")
    assert audit.unknown_keys == ("missing",)
    assert audit.unused_keys == ()
    assert audit.valid is False


def test_citation_audit_reports_unused_project_entries(tmp_path: Path) -> None:
    store = BibliographyStore(context(), tmp_path / "bibliography.jsonl")
    store.add(entry("known"))

    audit = store.audit_text("No citations in this text.")

    assert audit.used_keys == ()
    assert audit.unknown_keys == ()
    assert audit.unused_keys == ("known",)
    assert audit.valid is True
