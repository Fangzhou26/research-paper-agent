"""Safe project output writing with minimal provenance manifests."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from dataclasses import dataclass

from .bibliography import BibliographyStore
from .catalog import PaperCatalog
from .models import ArtifactRecord, SchemaValidationError, record_from_dict, record_to_dict
from .notes import EvidenceStore
from .project import ResearchContext


class OutputError(RuntimeError):
    """Raised when a project artifact cannot be validated or written."""


@dataclass(frozen=True)
class OutputContent:
    """A provenance-verified artifact and its text content."""

    artifact: ArtifactRecord
    content: str


class OutputWriter:
    """Write artifacts only inside the active project's output directory."""

    def __init__(
        self,
        context: ResearchContext,
        *,
        evidence_store: EvidenceStore | None = None,
        bibliography_store: BibliographyStore | None = None,
    ) -> None:
        self.context = context
        self.evidence_store = evidence_store or EvidenceStore(context)
        self.bibliography_store = bibliography_store or BibliographyStore(context)

    def write(
        self,
        filename: str | Path,
        content: str,
        *,
        paper_ids: tuple[str, ...] = (),
        evidence_ids: tuple[str, ...] = (),
        overwrite: bool = False,
        validate_citations: bool = True,
    ) -> ArtifactRecord:
        if not isinstance(content, str):
            raise OutputError("Output content must be text")
        target = self._target(filename)
        self._validate_dependencies(paper_ids, evidence_ids)
        if validate_citations:
            audit = self.bibliography_store.audit_text(content)
            if audit.unknown_keys:
                raise OutputError(
                    f"Output contains unknown citation keys: "
                    f"{', '.join(audit.unknown_keys)}"
                )
        if target.exists():
            try:
                existing_content = target.read_text(encoding="utf-8")
            except OSError as exc:
                raise OutputError(f"Cannot read existing output: {target}") from exc
            if existing_content != content and not overwrite:
                raise OutputError(f"Output already exists with different content: {target}")

        artifact = ArtifactRecord(
            artifact_id=_artifact_id(
                self.context.project_id,
                target,
                content,
                paper_ids,
                evidence_ids,
            ),
            project_id=self.context.project_id,
            artifact_type=target.suffix.lstrip(".") or "text",
            path=_portable_path(target, self.context.repository_root),
            paper_ids=paper_ids,
            evidence_ids=evidence_ids,
        )
        _write_text_atomic(target, content)
        manifest_path = target.with_name(f"{target.name}.manifest.json")
        manifest = json.dumps(
            record_to_dict(artifact),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _write_text_atomic(manifest_path, manifest)
        return artifact

    def read(self, artifact_id: str) -> OutputContent:
        """Read one output by its recorded artifact ID."""

        if not isinstance(artifact_id, str) or not artifact_id.strip():
            raise OutputError("Artifact ID must be a non-empty string")
        output_root = self.context.project.output_directory.resolve()
        manifests = sorted(output_root.rglob("*.manifest.json")) if output_root.exists() else ()
        for manifest_path in manifests:
            try:
                record = record_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError, SchemaValidationError) as exc:
                raise OutputError(f"Invalid output manifest: {manifest_path}") from exc
            if not isinstance(record, ArtifactRecord):
                raise OutputError(f"Output manifest does not contain an artifact: {manifest_path}")
            if record.artifact_id != artifact_id:
                continue
            if record.project_id != self.context.project_id:
                raise OutputError(f"Artifact belongs to another project: {artifact_id}")
            target = record.path
            if not target.is_absolute():
                target = self.context.repository_root / target
            target = target.resolve()
            if target == output_root or output_root not in target.parents:
                raise OutputError(f"Artifact path escapes output directory: {artifact_id}")
            try:
                content = target.read_text(encoding="utf-8")
            except OSError as exc:
                raise OutputError(f"Cannot read output artifact: {target}") from exc
            return OutputContent(record, content)
        raise OutputError(f"Unknown output artifact ID: {artifact_id}")

    def _target(self, filename: str | Path) -> Path:
        relative = Path(filename)
        if relative.is_absolute():
            raise OutputError("Output filename must be relative")
        output_root = self.context.project.output_directory.resolve()
        target = (output_root / relative).resolve()
        if target == output_root or output_root not in target.parents:
            raise OutputError("Output path escapes the project output directory")
        return target

    def _validate_dependencies(
        self,
        paper_ids: tuple[str, ...],
        evidence_ids: tuple[str, ...],
    ) -> None:
        catalog = PaperCatalog(self.context.catalog_path)
        unknown_papers = [paper_id for paper_id in paper_ids if catalog.get(paper_id) is None]
        if unknown_papers:
            raise OutputError(
                f"Unknown paper IDs in output provenance: {', '.join(unknown_papers)}"
            )
        unknown_evidence = [
            evidence_id
            for evidence_id in evidence_ids
            if self.evidence_store.get(evidence_id) is None
        ]
        if unknown_evidence:
            raise OutputError(
                f"Unknown evidence IDs in output provenance: "
                f"{', '.join(unknown_evidence)}"
            )


def _artifact_id(
    project_id: str,
    path: Path,
    content: str,
    paper_ids: tuple[str, ...],
    evidence_ids: tuple[str, ...],
) -> str:
    payload = json.dumps(
        {
            "project_id": project_id,
            "path": str(path),
            "content": content,
            "paper_ids": paper_ids,
            "evidence_ids": evidence_ids,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def _portable_path(path: Path, repository_root: Path) -> Path:
    try:
        return path.relative_to(repository_root)
    except ValueError:
        return path


def _write_text_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_path = Path(temporary.name)
        os.replace(temporary_path, path)
    except OSError as exc:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise OutputError(f"Cannot write output: {path}") from exc
