import json
from pathlib import Path

from research_agent.bibliography import BibliographyStore
from research_agent.catalog import PaperCatalog
from research_agent.discovery import (
    AcademicWork,
    ArxivProvider,
    DiscoveryQueryPlanner,
    OpenAlexProvider,
    SeedDiscoveryPipeline,
)
from research_agent.llm import LLMProviderError
from research_agent.ingest import ingest_paper
from research_agent.models import InnovationProfile
from research_agent.project import ResearchContext
from research_agent.retrieval import read_paper_pages


class StubLLM:
    def generate_json(self, messages):
        return {
            "queries": [
                {
                    "query": "perceptive humanoid motion tracking terrain",
                    "rationale": "Connect tracking and terrain perception",
                }
            ]
        }


class FailingLLM:
    def generate_json(self, messages):
        raise LLMProviderError("offline")


class StubAcademicProvider:
    def __init__(self) -> None:
        self.seed = AcademicWork(
            external_id="https://openalex.org/W1",
            title="Seed Tracking Paper",
            authors=("Ada Researcher",),
            year=2024,
            abstract="A general humanoid policy tracks diverse reference motions.",
            cited_by_count=20,
            topics=("Humanoid Robotics",),
        )
        self.related = AcademicWork(
            external_id="https://openalex.org/W2",
            title="Terrain Aware Tracking",
            authors=("Lin Scientist",),
            year=2025,
            abstract="Terrain perception adapts contacts while preserving motion intent.",
            doi="10.1000/terrain",
            cited_by_count=5,
            topics=("Robot Locomotion",),
        )
        self.metadata_only = AcademicWork(
            external_id="https://openalex.org/W3",
            title="Metadata Only Work",
            year=2026,
        )

    def neighbors(self, work, *, limit=8):
        return ((self.related, "cites_seed"), (self.metadata_only, "related_to_seed"))

    def search(self, query, *, limit=10):
        return (self.related, self.metadata_only)


def project(tmp_path: Path) -> ResearchContext:
    directory = tmp_path / "projects" / "demo"
    directory.mkdir(parents=True)
    (directory / "project.json").write_text(json.dumps({
        "title": "Demo",
        "research_question": "How do methods develop?",
        "description": None,
        "uploaded_papers": [],
        "seed_papers": [],
        "research_scope": [],
        "output_directory": "projects/demo/outputs",
    }), encoding="utf-8")
    return ResearchContext.load(tmp_path, "demo")


def profile(seed_paper_id: str = "sha256:seed") -> InnovationProfile:
    return InnovationProfile(
        project_id="demo",
        topic="Perceptive humanoid motion tracking",
        innovation_points=("Fuse raw motion intent with terrain observations",),
        seed_paper_ids=(seed_paper_id,),
    )


def test_seed_discovery_registers_screened_abstract_corpus(tmp_path: Path) -> None:
    context = project(tmp_path)
    seed_path = context.project_directory / "seed.pdf"
    seed_path.write_bytes(b"%PDF-1.4\n")
    seed = ingest_paper(
        seed_path,
        PaperCatalog(context.catalog_path),
        title="Seed Tracking Paper",
        path_base=context.repository_root,
    ).record
    events: list[str] = []
    result = SeedDiscoveryPipeline(
        context,
        StubAcademicProvider(),
        DiscoveryQueryPlanner(StubLLM()),
        on_event=lambda event_type, data: events.append(event_type),
    ).run(profile(seed.paper_id), max_papers=5)

    assert result.planner_mode == "llm"
    assert len(result.candidates) == 2
    assert len(result.accepted_paper_ids) == 2
    metadata = next(item for item in result.candidates if item.external_id.endswith("W3"))
    assert metadata.screening_decision == "rejected"
    assert "no abstract" in metadata.screening_reason
    assert len(PaperCatalog(context.catalog_path).all()) == 2
    assert len(BibliographyStore(context).all()) == 1
    pages = read_paper_pages(context, "openalex:W2", (1,))
    assert "Terrain perception" in pages[0].text
    assert result.report_path.is_file()
    assert events == [
        "discovery_started",
        "seed_loaded",
        "queries_planned",
        "search_completed",
        "screening_completed",
    ]


def test_query_planner_falls_back_when_model_is_unavailable() -> None:
    queries, mode = DiscoveryQueryPlanner(FailingLLM()).plan(
        profile(),
        ["Seed Tracking Paper"],
    )

    assert mode == "fallback"
    assert queries[0].query == profile().topic
    assert any("Fuse raw motion" in item.query for item in queries)


def test_openalex_adapter_reconstructs_abstract() -> None:
    def request(path, params):
        return {"results": [{
            "id": "https://openalex.org/W9",
            "display_name": "A Test Work",
            "authorships": [{"author": {"display_name": "A. Author"}}],
            "publication_year": 2025,
            "primary_location": {
                "landing_page_url": "https://example.test/work",
                "source": {"display_name": "Test Venue"},
            },
            "doi": "https://doi.org/10.1000/test",
            "abstract_inverted_index": {"Terrain": [0], "tracking": [1]},
            "cited_by_count": 4,
            "relevance_score": 10.0,
            "topics": [{"display_name": "Robotics"}],
            "related_works": [],
            "referenced_works": [],
        }]}

    work = OpenAlexProvider(request_json=request).search("terrain", limit=1)[0]

    assert work.title == "A Test Work"
    assert work.abstract == "Terrain tracking"
    assert work.doi == "10.1000/test"


def test_arxiv_adapter_parses_atom_abstract() -> None:
    xml = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <id>https://arxiv.org/abs/2601.01234v2</id>
        <title>Terrain Aware Tracking</title>
        <summary>We adapt motion to terrain observations.</summary>
        <published>2026-01-03T00:00:00Z</published>
        <author><name>Ada Researcher</name></author>
        <category term="cs.RO" />
      </entry>
    </feed>"""
    work = ArxivProvider(request_text=lambda params: xml).search("terrain", limit=1)[0]

    assert work.external_id == "ARXIV:2601.01234"
    assert work.year == 2026
    assert work.abstract == "We adapt motion to terrain observations."
