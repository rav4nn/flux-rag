"""Tests for fixed-size chunking strategy."""

import pytest

from fluxrag.chunking.fixed import FixedChunker
from fluxrag.core.schema import Document


chunker = FixedChunker()


def _make_doc(text: str) -> Document:
    return Document(text=text, metadata={"source_type": "txt", "source_path": "test.txt"})


def test_chunks_long_text() -> None:
    text = "Word " * 500  # ~2500 chars
    doc = _make_doc(text)
    chunks = chunker.chunk(doc, chunk_size=800)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 800
        assert chunk.document_id == doc.id
        assert chunk.metadata["chunking_strategy"] == "fixed"


def test_short_text_single_chunk() -> None:
    doc = _make_doc("A short sentence about coffee.")
    chunks = chunker.chunk(doc, chunk_size=800)
    assert len(chunks) == 1
    assert chunks[0].text == "A short sentence about coffee."


def test_minimal_text_single_chunk() -> None:
    doc = _make_doc("x")
    chunks = chunker.chunk(doc, chunk_size=800)
    assert len(chunks) == 1


def test_overlap_creates_more_chunks() -> None:
    text = "A" * 2000
    doc = _make_doc(text)
    chunks_no_overlap = chunker.chunk(doc, chunk_size=1000, overlap_ratio=0.0)
    chunks_with_overlap = chunker.chunk(doc, chunk_size=1000, overlap_ratio=0.2)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_chunk_ids_are_sequential() -> None:
    text = "Word " * 500
    doc = _make_doc(text)
    chunks = chunker.chunk(doc, chunk_size=200)
    for i, chunk in enumerate(chunks):
        assert chunk.index == i
        assert chunk.id == f"{doc.id}_chunk_{i}"
