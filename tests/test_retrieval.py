from pathlib import Path

import pytest

from research_agent import (
    ResearchContext,
    RetrievalError,
    list_papers,
    read_paper_pages,
    search_paper_pages,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def context() -> ResearchContext:
    return ResearchContext.load(REPOSITORY_ROOT, "demo")


def test_list_papers_uses_project_catalog() -> None:
    papers = list_papers(context())

    assert len(papers) == 1
    assert all(paper.paper_id.startswith("synthetic:") for paper in papers)


def test_read_pages_preserves_requested_order() -> None:
    paper = list_papers(context())[0]

    pages = read_paper_pages(context(), paper.paper_id, [2, 1])

    assert [page.page for page in pages] == [2, 1]
    assert all(page.paper_id == paper.paper_id for page in pages)


def test_search_pages_returns_ranked_explainable_matches() -> None:
    matches = search_paper_pages(context(), "terrain reconstruction", limit=5)

    assert matches
    assert len(matches) <= 5
    assert [match.score for match in matches] == sorted(
        (match.score for match in matches), reverse=True
    )
    assert all(match.page >= 1 and match.snippet for match in matches)


def test_search_pages_can_limit_search_to_selected_paper() -> None:
    paper = list_papers(context())[0]

    matches = search_paper_pages(
        context(),
        "motion",
        paper_ids=[paper.paper_id],
    )

    assert all(match.paper_id == paper.paper_id for match in matches)


@pytest.mark.parametrize("query", ["", "   "])
def test_search_rejects_empty_query(query: str) -> None:
    with pytest.raises(RetrievalError, match="non-empty"):
        search_paper_pages(context(), query)


def test_read_rejects_page_outside_paper() -> None:
    paper = list_papers(context())[0]

    with pytest.raises(RetrievalError, match="outside paper range"):
        read_paper_pages(context(), paper.paper_id, [9999])


def test_search_rejects_unknown_paper_filter() -> None:
    with pytest.raises(RetrievalError, match="Unknown paper IDs"):
        search_paper_pages(context(), "motion", paper_ids=["missing"])
