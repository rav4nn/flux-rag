"""Data models for the FluxRAG pipeline."""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, field_validator, model_validator


class Document(BaseModel):
    """Unified document schema. All parsers produce this."""

    id: str = ""
    text: str
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _auto_id(self) -> Document:
        if not self.id:
            source = self.metadata.get("source_path", "")
            hash_input = f"{source}:{self.text[:500]}"
            self.id = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
        return self

    @field_validator("text")
    @classmethod
    def _text_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Document text must not be empty")
        return v


class Chunk(BaseModel):
    """A chunk of a document, produced by a chunking strategy."""

    id: str = ""
    text: str
    document_id: str
    index: int
    metadata: dict[str, Any] = {}

    @model_validator(mode="after")
    def _auto_id(self) -> Chunk:
        if not self.id:
            self.id = f"{self.document_id}_chunk_{self.index}"
        return self


class EmbeddedChunk(BaseModel):
    """A chunk with its embedding vector."""

    chunk: Chunk
    embedding: list[float]


class RetrievalResult(BaseModel):
    """A single result from retrieval."""

    chunk: Chunk
    score: float


class RerankedResult(BaseModel):
    """A result after reranking."""

    chunk: Chunk
    score: float
    original_rank: int


class QueryResult(BaseModel):
    """Full response from a pipeline query."""

    answer: str
    sources: list[RetrievalResult]
    latency_ms: float


class QAPair(BaseModel):
    """A question-answer pair for evaluation."""

    question: str
    ground_truth: str
    source_chunk_id: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"


class EvalResult(BaseModel):
    """RAGAS evaluation scores."""

    context_recall: float
    context_precision: float
    faithfulness: float
    answer_relevancy: float

    def summary(self) -> str:
        return (
            f"Context Recall: {self.context_recall:.3f} | "
            f"Context Precision: {self.context_precision:.3f} | "
            f"Faithfulness: {self.faithfulness:.3f} | "
            f"Answer Relevancy: {self.answer_relevancy:.3f}"
        )


class BenchmarkResult(BaseModel):
    """Results from a benchmark matrix run."""

    matrix_name: str
    rows: list[dict[str, Any]]

    def summary(self) -> str:
        return f"Matrix {self.matrix_name}: {len(self.rows)} configurations evaluated"
