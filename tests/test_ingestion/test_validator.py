"""Tests for document validator."""

import pytest

from fluxrag.core.schema import Document
from fluxrag.ingestion.validator import DocumentValidator


validator = DocumentValidator()


def test_validates_good_document() -> None:
    doc = Document(
        text="Valid document text with enough content",
        metadata={"source_type": "txt", "source_path": "test.txt"},
    )
    result = validator.validate(doc)
    assert result.id == doc.id


def test_rejects_short_text() -> None:
    doc = Document(
        text="Short",
        metadata={"source_type": "txt"},
    )
    with pytest.raises(ValueError, match="too short"):
        validator.validate(doc)


def test_rejects_missing_source_type() -> None:
    doc = Document(
        text="This document has enough text content",
        metadata={"source_path": "test.txt"},
    )
    with pytest.raises(ValueError, match="source_type"):
        validator.validate(doc)


def test_batch_filters_invalid() -> None:
    docs = [
        Document(text="Valid document text with source type", metadata={"source_type": "txt"}),
        Document(text="Short", metadata={"source_type": "txt"}),
        Document(text="Another valid document text here", metadata={"source_type": "pdf"}),
    ]
    valid = validator.validate_batch(docs)
    assert len(valid) == 2
