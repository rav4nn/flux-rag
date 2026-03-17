"""Tests for JSONL parser."""

from pathlib import Path

from fluxrag.ingestion.jsonl import JSONLParser


parser = JSONLParser()


def test_parse_jsonl(sample_jsonl: Path) -> None:
    docs = parser.parse(str(sample_jsonl))
    assert len(docs) == 2  # 2 valid, 1 empty line, 1 empty text
    assert docs[0].id == "doc_1"
    assert "Grind size" in docs[0].text


def test_empty_jsonl(empty_jsonl: Path) -> None:
    docs = parser.parse(str(empty_jsonl))
    assert docs == []


def test_preserves_metadata(tmp_dir: Path) -> None:
    import json

    p = tmp_dir / "meta.jsonl"
    p.write_text(
        json.dumps({
            "id": "x",
            "text": "Some valid text content here",
            "metadata": {"source_type": "custom", "brew_method": "espresso"},
        }),
        encoding="utf-8",
    )
    docs = parser.parse(str(p))
    assert docs[0].metadata["brew_method"] == "espresso"
    assert docs[0].metadata["source_type"] == "custom"


def test_skips_malformed_lines(tmp_dir: Path) -> None:
    p = tmp_dir / "bad.jsonl"
    p.write_text('not json\n{"text": "Valid line here for testing"}\n', encoding="utf-8")
    docs = parser.parse(str(p))
    assert len(docs) == 1
