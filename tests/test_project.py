import json
from pathlib import Path
import pytest

from research_agent import (
    BibliographyStore,
    PaperCatalog,
    ProjectConfigurationError,
    ResearchContext,
    load_extracted_pages,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_active_project_uses_public_synthetic_demo() -> None:
    context = ResearchContext.load_active(REPOSITORY_ROOT)

    assert context.project_id == "demo"
    assert context.project.title == "Research Agent Synthetic Demo"
    assert context.project.output_directory == (REPOSITORY_ROOT / "projects/demo/outputs")
    assert context.project.uploaded_papers == ()
    assert context.missing_uploaded_papers() == ()
    assert context.brief_path.is_file()
    assert context.style_path.is_file()


def test_demo_fixture_is_explicitly_synthetic() -> None:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    assert "fictional" in (context.project.description or "").casefold()
    assert context.project.uploaded_papers == ()


def test_demo_catalog_and_bibliography_are_self_contained() -> None:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    catalog = PaperCatalog(context.catalog_path)
    records = catalog.all()
    bibliography = BibliographyStore(context)

    assert len(records) == 1
    assert records[0].file_path is None
    assert len(bibliography.all()) == 1
    assert bibliography.for_paper(records[0].paper_id) is not None


def test_demo_fixture_has_page_cache_for_cataloged_paper() -> None:
    context = ResearchContext.load(REPOSITORY_ROOT, "demo")
    catalog = PaperCatalog(context.project_directory / "catalog.jsonl")
    extraction_directory = context.project_directory / "extracted"

    for record in catalog.all():
        safe_id = record.paper_id.replace(":", "_")
        pages = load_extracted_pages(
            extraction_directory / f"{safe_id}.pages.jsonl",
            expected_paper_id=record.paper_id,
        )
        assert pages
        assert "synthetic" in pages[0].text.casefold()


def test_arbitrary_project_has_no_topic_or_legacy_path_dependency(tmp_path: Path) -> None:
    paper = tmp_path / "uploads" / "paper.pdf"
    paper.parent.mkdir()
    paper.write_bytes(b"%PDF-placeholder")
    project_dir = tmp_path / "projects" / "climate-risk"
    project_dir.mkdir(parents=True)
    config = {
        "title": "Climate Risk",
        "research_question": "How are compound climate risks modeled?",
        "description": None,
        "uploaded_papers": ["uploads/paper.pdf"],
        "seed_papers": ["doi:10.0000/example"],
        "research_scope": ["compound events", "risk modeling"],
        "output_directory": "projects/climate-risk/outputs",
    }
    (project_dir / "project.json").write_text(json.dumps(config), encoding="utf-8")

    context = ResearchContext.load(tmp_path, "climate-risk")

    assert context.project.research_question == config["research_question"]
    assert context.project.uploaded_papers == (paper.resolve(),)
    assert context.project.seed_papers == ("doi:10.0000/example",)
    assert context.project.research_scope == ("compound events", "risk modeling")
    assert context.project.output_directory == (project_dir / "outputs").resolve()


@pytest.mark.parametrize("missing_key", ["title", "research_question", "output_directory"])
def test_required_fields_are_validated(tmp_path: Path, missing_key: str) -> None:
    project_dir = tmp_path / "projects" / "invalid"
    project_dir.mkdir(parents=True)
    config = {
        "title": "Title",
        "research_question": "Question?",
        "output_directory": "outputs",
    }
    del config[missing_key]
    (project_dir / "project.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ProjectConfigurationError, match=missing_key):
        ResearchContext.load(tmp_path, "invalid")


def test_project_id_cannot_escape_projects_directory(tmp_path: Path) -> None:
    with pytest.raises(ProjectConfigurationError, match="Invalid project id"):
        ResearchContext.load(tmp_path, "../outside")
