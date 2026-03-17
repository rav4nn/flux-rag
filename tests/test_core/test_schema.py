"""Tests for FluxRAG data models."""

import pytest

from fluxrag.core.schema import Chunk, Document, EvalResult


def test_document_auto_generates_id() -> None:
    doc = Document(text="Some content", metadata={"source_path": "test.txt"})
    assert doc.id != ""
    assert len(doc.id) == 16


def test_document_preserves_explicit_id() -> None:
    doc = Document(id="custom_id", text="Some content")
    assert doc.id == "custom_id"


def test_document_rejects_empty_text() -> None:
    with pytest.raises(ValueError, match="empty"):
        Document(text="   ")


def test_chunk_auto_generates_id() -> None:
    chunk = Chunk(text="chunk text", document_id="doc_123", index=0)
    assert chunk.id == "doc_123_chunk_0"


def test_eval_result_summary() -> None:
    result = EvalResult(
        context_recall=0.85,
        context_precision=0.78,
        faithfulness=0.92,
        answer_relevancy=0.88,
    )
    summary = result.summary()
    assert "0.850" in summary
    assert "0.780" in summary
