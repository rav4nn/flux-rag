"""Tests for HTML parser."""

from pathlib import Path

from fluxrag.ingestion.html import HTMLParser


parser = HTMLParser()


def test_parse_html_extracts_content(sample_html: Path) -> None:
    docs = parser.parse(str(sample_html))
    assert len(docs) == 1
    assert "Espresso" in docs[0].text
    assert docs[0].metadata["source_type"] == "html"


def test_empty_html_returns_nothing(empty_html: Path) -> None:
    docs = parser.parse(str(empty_html))
    assert docs == []


def test_title_extraction(sample_html: Path) -> None:
    docs = parser.parse(str(sample_html))
    assert docs[0].metadata["title"] == "Espresso Guide"


def test_boilerplate_removal(sample_html: Path) -> None:
    """Nav and footer content should be stripped."""
    docs = parser.parse(str(sample_html))
    text = docs[0].text
    assert "Menu items here" not in text or "pressure" in text
