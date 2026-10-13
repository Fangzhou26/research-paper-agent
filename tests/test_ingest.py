import hashlib
from pathlib import Path

import pytest

from research_agent import IngestionError, PaperCatalog, ingest_paper


PDF_BYTES = b"%PDF-1.7\nminimal test content\n%%EOF\n"


def write_pdf(path: Path, content: bytes = PDF_BYTES) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_ingest_registers_pdf_with_content_hash_id(tmp_path: Path) -> None:
    source = write_pdf(tmp_path / "uploads" / "paper.pdf")
    catalog = PaperCatalog(tmp_path / "catalog.jsonl")

    result = ingest_paper(
        source,
        catalog,
        title="Verified title",
        authors=("An Author",),
        year=2026,
        arxiv_id="2601.00001",
        tags=("example",),
    )

    expected_hash = hashlib.sha256(PDF_BYTES).hexdigest()
    assert result.created is True
    assert result.record.paper_id == f"sha256:{expected_hash}"
    assert result.record.file_path == source.resolve()
    assert result.record.title == "Verified title"
    assert catalog.get(result.record.paper_id) == result.record


def test_ingest_deduplicates_same_content_at_different_paths(tmp_path: Path) -> None:
    first = write_pdf(tmp_path / "first.pdf")
    second = write_pdf(tmp_path / "second-name.pdf")
    catalog = PaperCatalog(tmp_path / "catalog.jsonl")

    first_result = ingest_paper(first, catalog, title="Canonical title")
    duplicate_result = ingest_paper(second, catalog, title="Ignored duplicate title")

    assert first_result.created is True
    assert duplicate_result.created is False
    assert duplicate_result.record == first_result.record
    assert len(catalog.all()) == 1


def test_ingest_uses_filename_as_unverified_fallback_title(tmp_path: Path) -> None:
    source = write_pdf(tmp_path / "descriptive-filename.pdf")

    result = ingest_paper(source, PaperCatalog(tmp_path / "catalog.jsonl"))

    assert result.record.title == "descriptive-filename"
    assert result.record.verification_status == "unverified"


def test_ingest_can_store_portable_path_relative_to_base(tmp_path: Path) -> None:
    source = write_pdf(tmp_path / "projects" / "example" / "papers" / "paper.pdf")

    result = ingest_paper(
        source,
        PaperCatalog(tmp_path / "catalog.jsonl"),
        path_base=tmp_path,
    )

    expected_path = Path("projects/example/papers/paper.pdf")
    assert result.record.file_path == expected_path
    assert result.record.source_uri == str(expected_path)


def test_ingest_rejects_source_outside_path_base(tmp_path: Path) -> None:
    source = write_pdf(tmp_path / "paper.pdf")
    unrelated_base = tmp_path / "unrelated"
    unrelated_base.mkdir()

    with pytest.raises(IngestionError, match="outside the configured path base"):
        ingest_paper(
            source,
            PaperCatalog(tmp_path / "catalog.jsonl"),
            path_base=unrelated_base,
        )


@pytest.mark.parametrize(
    ("filename", "content", "message"),
    [
        ("not-a-paper.txt", b"plain text", "not a PDF"),
        ("empty.pdf", b"", "not a PDF"),
    ],
)
def test_ingest_rejects_non_pdf_content(
    tmp_path: Path, filename: str, content: bytes, message: str
) -> None:
    source = tmp_path / filename
    source.write_bytes(content)

    with pytest.raises(IngestionError, match=message):
        ingest_paper(source, PaperCatalog(tmp_path / "catalog.jsonl"))


def test_ingest_rejects_missing_source(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="does not exist"):
        ingest_paper(
            tmp_path / "missing.pdf",
            PaperCatalog(tmp_path / "catalog.jsonl"),
        )
