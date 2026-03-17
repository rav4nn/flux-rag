"""Tests for PDF parser."""

from pathlib import Path

import pymupdf
import pytest

from fluxrag.ingestion.pdf import PDFParser


parser = PDFParser()


@pytest.fixture
def sample_pdf(tmp_dir: Path) -> Path:
    """Create a minimal PDF with text content."""
    p = tmp_dir / "sample.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Coffee extraction is the process of dissolving soluble compounds from "
        "ground coffee into water. The ideal extraction yield is between 18-22%. "
        "Under-extraction produces sour, acidic flavors while over-extraction "
        "leads to bitterness and astringency.",
    )
    doc.save(str(p))
    doc.close()
    return p


@pytest.fixture
def empty_pdf(tmp_dir: Path) -> Path:
    """Create a PDF with no text."""
    p = tmp_dir / "empty.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(str(p))
    doc.close()
    return p


def test_parse_pdf(sample_pdf: Path) -> None:
    docs = parser.parse(str(sample_pdf))
    assert len(docs) == 1
    assert "extraction" in docs[0].text
    assert docs[0].metadata["source_type"] == "pdf"
    assert docs[0].metadata["page_count"] == 1


def test_empty_pdf_returns_nothing(empty_pdf: Path) -> None:
    docs = parser.parse(str(empty_pdf))
    assert docs == []


def test_multipage_pdf(tmp_dir: Path) -> None:
    p = tmp_dir / "multi.pdf"
    doc = pymupdf.open()
    for i in range(3):
        page = doc.new_page()
        page.insert_text(
            (72, 72),
            f"Page {i+1}: This is a detailed paragraph about coffee brewing "
            f"that contains enough text to pass the minimum length threshold.",
        )
    doc.save(str(p))
    doc.close()

    docs = parser.parse(str(p))
    assert len(docs) == 1
    assert docs[0].metadata["page_count"] == 3
    assert "Page 1" in docs[0].text
    assert "Page 3" in docs[0].text
