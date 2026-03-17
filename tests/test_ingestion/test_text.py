"""Tests for text and Markdown parser."""

from pathlib import Path

from fluxrag.ingestion.text import TextParser


parser = TextParser()


def test_parse_txt(sample_txt: Path) -> None:
    docs = parser.parse(str(sample_txt))
    assert len(docs) == 1
    assert "extraction" in docs[0].text
    assert docs[0].metadata["source_type"] == "txt"


def test_parse_markdown(sample_md: Path) -> None:
    docs = parser.parse(str(sample_md))
    assert len(docs) == 1
    assert "V60" in docs[0].text
    assert docs[0].metadata["source_type"] == "md"


def test_empty_file_returns_nothing(empty_txt: Path) -> None:
    docs = parser.parse(str(empty_txt))
    assert docs == []


def test_minimal_content(minimal_txt: Path) -> None:
    docs = parser.parse(str(minimal_txt))
    assert len(docs) == 1
    assert docs[0].text == "Hello"


def test_supported_extensions() -> None:
    assert ".txt" in parser.supported_extensions()
    assert ".md" in parser.supported_extensions()
