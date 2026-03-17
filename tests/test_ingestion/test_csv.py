"""Tests for CSV/TSV parser."""

from pathlib import Path

from fluxrag.ingestion.csv_parser import CSVParser


parser = CSVParser()


def test_parse_csv(sample_csv: Path) -> None:
    docs = parser.parse(str(sample_csv))
    assert len(docs) == 1
    assert "Yirgacheffe" in docs[0].text
    assert docs[0].metadata["source_type"] == "csv"


def test_empty_csv_returns_nothing(empty_csv: Path) -> None:
    docs = parser.parse(str(empty_csv))
    assert docs == []


def test_csv_row_grouping(tmp_dir: Path) -> None:
    """With rows_per_doc=1, each row should become its own document."""
    p = tmp_dir / "multi.csv"
    p.write_text("col\nrow1\nrow2\nrow3\n", encoding="utf-8")
    docs = parser.parse(str(p), rows_per_doc=1)
    assert len(docs) == 3


def test_tsv_support(tmp_dir: Path) -> None:
    p = tmp_dir / "data.tsv"
    p.write_text("name\torigin\nGesha\tPanama\n", encoding="utf-8")
    docs = parser.parse(str(p))
    assert len(docs) == 1
    assert "Gesha" in docs[0].text
