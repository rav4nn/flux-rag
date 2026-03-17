"""Tests for the parser router."""

from pathlib import Path

import pytest

from fluxrag.ingestion.router import ParserRouter, create_default_router


def test_default_router_registers_all_parsers() -> None:
    router = create_default_router()
    exts = router.supported_extensions
    assert ".txt" in exts
    assert ".pdf" in exts
    assert ".docx" in exts
    assert ".csv" in exts
    assert ".html" in exts
    assert ".jsonl" in exts
    assert ".png" in exts
    assert ".mp3" in exts
    assert ".youtube" in exts


def test_router_dispatches_txt(sample_txt: Path) -> None:
    router = create_default_router()
    docs = router.parse(str(sample_txt))
    assert len(docs) == 1
    assert docs[0].metadata["source_type"] == "txt"


def test_router_rejects_unknown_extension(tmp_dir: Path) -> None:
    router = create_default_router()
    p = tmp_dir / "file.xyz"
    p.write_text("content", encoding="utf-8")
    with pytest.raises(ValueError, match="No parser registered"):
        router.parse(str(p))
