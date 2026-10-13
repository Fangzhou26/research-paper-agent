import json
from pathlib import Path

import pytest

from research_agent import (
    CatalogConflictError,
    CatalogError,
    PageText,
    PaperCatalog,
    PaperRecord,
    ingest_paper,
    record_to_dict,
)


def paper_record(paper_id: str = "sha256:abc", title: str = "Paper") -> PaperRecord:
    return PaperRecord(
        paper_id=paper_id,
        title=title,
        authors=("An Author",),
        file_path=Path("papers/source.pdf"),
    )


def test_catalog_add_and_load_round_trip(tmp_path: Path) -> None:
    catalog_path = tmp_path / "library" / "catalog.jsonl"
    catalog = PaperCatalog(catalog_path)
    record = paper_record()

    result = catalog.add(record)

    assert result.created is True
    assert catalog_path.is_file()
    assert PaperCatalog(catalog_path).all() == (record,)
    assert PaperCatalog(catalog_path).get(record.paper_id) == record


def test_catalog_readding_identical_record_is_idempotent(tmp_path: Path) -> None:
    catalog = PaperCatalog(tmp_path / "catalog.jsonl")
    record = paper_record()

    assert catalog.add(record).created is True
    assert catalog.add(record).created is False
    assert len(catalog.all()) == 1


def test_catalog_rejects_conflicting_record_for_same_id(tmp_path: Path) -> None:
    catalog = PaperCatalog(tmp_path / "catalog.jsonl")
    catalog.add(paper_record(title="Original title"))

    with pytest.raises(CatalogConflictError, match="different metadata"):
        catalog.add(paper_record(title="Conflicting title"))


def test_catalog_rejects_non_paper_records(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    page = PageText(
        paper_id="paper-1",
        page=1,
        text="Text",
        extraction_method="native",
    )
    catalog_path.write_text(json.dumps(record_to_dict(page)) + "\n", encoding="utf-8")

    with pytest.raises(CatalogError, match="Non-paper record"):
        PaperCatalog(catalog_path).all()


def test_catalog_reports_invalid_json_line(tmp_path: Path) -> None:
    catalog_path = tmp_path / "catalog.jsonl"
    catalog_path.write_text("{not-json}\n", encoding="utf-8")

    with pytest.raises(CatalogError, match=":1"):
        PaperCatalog(catalog_path).all()

