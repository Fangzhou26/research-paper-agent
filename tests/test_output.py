from dataclasses import replace
import json
from pathlib import Path

import pytest

from research_agent import (
    ArtifactRecord,
    OutputError,
    OutputWriter,
    ResearchContext,
    list_papers,
    record_from_dict,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def context_with_output(output_directory: Path) -> ResearchContext:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    project = replace(context.project, output_directory=output_directory)
    return replace(context, project=project)


def test_output_writer_confines_file_and_writes_manifest(tmp_path: Path) -> None:
    context = context_with_output(tmp_path / "outputs")
    paper = list_papers(context)[0]
    writer = OutputWriter(context)

    artifact = writer.write(
        "drafts/summary.tex",
        r"A supported statement \cite{synthetic2025}.",
        paper_ids=(paper.paper_id,),
    )

    target = context.project.output_directory / "drafts/summary.tex"
    manifest_path = target.with_name("summary.tex.manifest.json")
    manifest = record_from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
    assert target.read_text(encoding="utf-8") == r"A supported statement \cite{synthetic2025}."
    assert isinstance(manifest, ArtifactRecord)
    assert manifest == artifact
    assert artifact.paper_ids == (paper.paper_id,)

    loaded = writer.read(artifact.artifact_id)
    assert loaded.artifact == artifact
    assert loaded.content == r"A supported statement \cite{synthetic2025}."


def test_output_reader_rejects_unknown_artifact_id(tmp_path: Path) -> None:
    writer = OutputWriter(context_with_output(tmp_path / "outputs"))

    with pytest.raises(OutputError, match="Unknown output artifact ID"):
        writer.read("sha256:unknown")


@pytest.mark.parametrize("filename", ["../outside.md", "/tmp/outside.md", "."])
def test_output_writer_rejects_paths_outside_output_directory(
    tmp_path: Path, filename: str
) -> None:
    writer = OutputWriter(context_with_output(tmp_path / "outputs"))

    with pytest.raises(OutputError, match="relative|escapes"):
        writer.write(filename, "content")


def test_output_writer_rejects_unknown_citation_key(tmp_path: Path) -> None:
    writer = OutputWriter(context_with_output(tmp_path / "outputs"))

    with pytest.raises(OutputError, match="unknown citation keys"):
        writer.write("draft.tex", r"Unsupported \cite{not_in_bibliography}.")


def test_output_writer_requires_explicit_overwrite(tmp_path: Path) -> None:
    writer = OutputWriter(context_with_output(tmp_path / "outputs"))
    writer.write("draft.md", "first")

    with pytest.raises(OutputError, match="already exists"):
        writer.write("draft.md", "second")

    artifact = writer.write("draft.md", "second", overwrite=True)
    assert artifact.path.name == "draft.md"
    assert (tmp_path / "outputs/draft.md").read_text(encoding="utf-8") == "second"


def test_output_writer_rejects_unknown_provenance_paper(tmp_path: Path) -> None:
    writer = OutputWriter(context_with_output(tmp_path / "outputs"))

    with pytest.raises(OutputError, match="Unknown paper IDs"):
        writer.write("draft.md", "content", paper_ids=("missing-paper",))
