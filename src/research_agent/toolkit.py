"""Minimal project-aware tool surface for a future single-agent loop."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from .bibliography import BibliographyStore, CitationAudit
from .models import ArtifactRecord, EvidenceRecord, PageText, PaperNote, PaperRecord
from .notes import EvidenceStore, PaperNoteStore, StoreWriteResult
from .output import OutputContent, OutputWriter
from .project import ResearchContext
from .retrieval import PageMatch, list_papers, read_paper_pages, search_paper_pages


class ResearchToolkit:
    """Cohesive tools required by the first research-agent implementation."""

    def __init__(
        self,
        context: ResearchContext,
        *,
        evidence_store: EvidenceStore | None = None,
        note_store: PaperNoteStore | None = None,
        bibliography_store: BibliographyStore | None = None,
        output_writer: OutputWriter | None = None,
    ) -> None:
        self.context = context
        self.evidence_store = evidence_store or EvidenceStore(context)
        self.note_store = note_store or PaperNoteStore(
            context,
            evidence_store=self.evidence_store,
        )
        self.bibliography_store = bibliography_store or BibliographyStore(context)
        self.output_writer = output_writer or OutputWriter(
            context,
            evidence_store=self.evidence_store,
            bibliography_store=self.bibliography_store,
        )

    def list_papers(self) -> tuple[PaperRecord, ...]:
        return list_papers(self.context)

    def search_pages(
        self,
        query: str,
        *,
        paper_ids: Iterable[str] | None = None,
        limit: int = 20,
    ) -> tuple[PageMatch, ...]:
        return search_paper_pages(
            self.context,
            query,
            paper_ids=paper_ids,
            limit=limit,
        )

    def read_pages(
        self,
        paper_id: str,
        pages: Iterable[int],
    ) -> tuple[PageText, ...]:
        return read_paper_pages(self.context, paper_id, pages)

    def add_evidence(
        self,
        paper_id: str,
        page: int,
        passage: str,
        interpretation: str,
        evidence_type: str,
        *,
        support_type: str = "direct",
    ) -> EvidenceRecord:
        evidence = EvidenceRecord(
            evidence_id=_evidence_id(
                self.context.project_id,
                paper_id,
                page,
                passage,
                interpretation,
                evidence_type,
                support_type,
            ),
            project_id=self.context.project_id,
            paper_id=paper_id,
            page=page,
            passage=passage,
            interpretation=interpretation,
            evidence_type=evidence_type,
            support_type=support_type,
        )
        self.evidence_store.add(evidence)
        return evidence

    def list_evidence(
        self,
        *,
        paper_id: str | None = None,
    ) -> tuple[EvidenceRecord, ...]:
        return self.evidence_store.list(paper_id=paper_id)

    def save_note(
        self,
        paper_id: str,
        research_problem: str,
        method: str,
        *,
        data_or_inputs: tuple[str, ...] = (),
        key_contributions: tuple[str, ...] = (),
        main_results: tuple[str, ...] = (),
        limitations: tuple[str, ...] = (),
        relevance: str | None = None,
        topics: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        uncertainties: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> tuple[PaperNote, StoreWriteResult]:
        note = PaperNote(
            project_id=self.context.project_id,
            paper_id=paper_id,
            research_problem=research_problem,
            method=method,
            data_or_inputs=data_or_inputs,
            key_contributions=key_contributions,
            main_results=main_results,
            limitations=limitations,
            relevance=relevance,
            topics=topics,
            evidence_ids=evidence_ids,
            uncertainties=uncertainties,
        )
        return note, self.note_store.save(note, overwrite=overwrite)

    def load_note(self, paper_id: str) -> PaperNote | None:
        return self.note_store.load(paper_id)

    def citation_key(self, paper_id: str) -> str:
        return self.bibliography_store.citation_key_for(paper_id)

    def audit_citations(self, text: str) -> CitationAudit:
        return self.bibliography_store.audit_text(text)

    def write_output(
        self,
        filename: str | Path,
        content: str,
        *,
        paper_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        overwrite: bool = False,
    ) -> ArtifactRecord:
        return self.output_writer.write(
            filename,
            content,
            paper_ids=paper_ids,
            evidence_ids=evidence_ids,
            overwrite=overwrite,
        )

    def read_output(self, artifact_id: str) -> OutputContent:
        return self.output_writer.read(artifact_id)


def _evidence_id(
    project_id: str,
    paper_id: str,
    page: int,
    passage: str,
    interpretation: str,
    evidence_type: str,
    support_type: str,
) -> str:
    payload = json.dumps(
        {
            "project_id": project_id,
            "paper_id": paper_id,
            "page": page,
            "passage": passage,
            "interpretation": interpretation,
            "evidence_type": evidence_type,
            "support_type": support_type,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"
