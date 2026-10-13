import json
from pathlib import Path

import pytest

from research_agent import (
    ExtractionError,
    PageText,
    PaperCatalog,
    PaperRecord,
    extract_catalog_paper,
    extract_paper,
    load_extracted_pages,
    record_to_dict,
)


class FakeBackend:
    name = "fake"

    def __init__(self, pages: tuple[str, ...]) -> None:
        self.pages = pages
        self.calls = 0

    def extract_pages(self, source_path: Path) -> tuple[str, ...]:
        assert source_path.is_file()
        self.calls += 1
        return self.pages


def paper(path: Path, paper_id: str = "sha256:abc") -> PaperRecord:
    return PaperRecord(paper_id=paper_id, title="Paper", file_path=path)


def test_extract_writes_page_addressable_cache(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    backend = FakeBackend(("A" * 120, "", "short"))

    result = extract_paper(
        paper(source),
        tmp_path / "extracted",
        backend=backend,
    )
    pages = load_extracted_pages(result.output_path, expected_paper_id="sha256:abc")

    assert result.created is True
    assert result.page_count == 3
    assert result.empty_pages == (2,)
    assert [page.page for page in pages] == [1, 2, 3]
    assert [page.quality for page in pages] == ["good", "empty", "low"]
    assert all(page.extraction_method == "fake" for page in pages)


def test_extract_reuses_valid_cache_unless_forced(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\n")
    backend = FakeBackend(("cached text",))
    record = paper(source)
    output_directory = tmp_path / "extracted"

    first = extract_paper(record, output_directory, backend=backend)
    cached = extract_paper(record, output_directory, backend=backend)
    forced = extract_paper(record, output_directory, backend=backend, force=True)

    assert first.created is True
    assert cached.created is False
    assert forced.created is True
    assert backend.calls == 2


def test_extract_catalog_paper_resolves_workspace_relative_path(tmp_path: Path) -> None:
    source = tmp_path / "projects" / "example" / "papers" / "paper.pdf"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"%PDF-1.7\n")
    catalog = PaperCatalog(tmp_path / "projects/example/catalog.jsonl")
    catalog.add(paper(source.relative_to(tmp_path)))

    result = extract_catalog_paper(
        catalog,
        "sha256:abc",
        tmp_path / "projects/example/extracted",
        path_base=tmp_path,
        backend=FakeBackend(("page text",)),
    )

    assert result.page_count == 1
    assert result.output_path.parent.name == "extracted"


def test_extract_rejects_unknown_catalog_paper(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="not present in catalog"):
        extract_catalog_paper(
            PaperCatalog(tmp_path / "catalog.jsonl"),
            "missing",
            tmp_path / "extracted",
        )


def test_extract_rejects_relative_path_without_base(tmp_path: Path) -> None:
    with pytest.raises(ExtractionError, match="requires path_base"):
        extract_paper(
            paper(Path("papers/paper.pdf")),
            tmp_path / "extracted",
            backend=FakeBackend(("text",)),
        )


def test_extract_rejects_pdf_with_no_pages(tmp_path: Path) -> None:
    source = tmp_path / "paper.pdf"
    source.write_bytes(b"%PDF-1.7\n")

    with pytest.raises(ExtractionError, match="contains no pages"):
        extract_paper(
            paper(source),
            tmp_path / "extracted",
            backend=FakeBackend(()),
        )


def test_load_rejects_non_contiguous_page_cache(tmp_path: Path) -> None:
    cache = tmp_path / "pages.jsonl"
    page = PageText(
        paper_id="paper-1",
        page=2,
        text="text",
        extraction_method="fake",
    )
    cache.write_text(json.dumps(record_to_dict(page)) + "\n", encoding="utf-8")

    with pytest.raises(ExtractionError, match="non-contiguous"):
        load_extracted_pages(cache)


def test_load_rejects_cache_for_different_paper(tmp_path: Path) -> None:
    cache = tmp_path / "pages.jsonl"
    page = PageText(
        paper_id="paper-1",
        page=1,
        text="text",
        extraction_method="fake",
    )
    cache.write_text(json.dumps(record_to_dict(page)) + "\n", encoding="utf-8")

    with pytest.raises(ExtractionError, match="does not match"):
        load_extracted_pages(cache, expected_paper_id="paper-2")
