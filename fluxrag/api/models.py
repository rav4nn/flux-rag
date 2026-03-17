"""Request/response Pydantic models for the FluxRAG API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class QueryRequest(BaseModel):
    question: str
    filters: dict[str, Any] | None = None
    top_k: int = 5


class SourceResponse(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any]
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    corpus_size: int
    index_ready: bool


class BenchmarkRunRequest(BaseModel):
    matrices: list[str]
    budget_limit_usd: float = 20.0


class BenchmarkRunResponse(BaseModel):
    job_id: str
    status: str
