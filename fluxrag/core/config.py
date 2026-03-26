"""Pydantic configuration models that validate domain.yaml."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, field_validator


class DomainConfig(BaseModel):
    name: str
    description: str


class CorpusSource(BaseModel):
    path: str
    types: list[str] | None = None
    type: str | None = None
    weight: float = 1.0
    transcription_model: str | None = None
    ocr_engine: str | None = None


class CorpusConfig(BaseModel):
    sources: list[CorpusSource]
    dedup_key: str = "id"


class ChunkingConfig(BaseModel):
    strategy: Literal["fixed", "sentence", "semantic"] = "fixed"
    target_tokens: int = 200
    overlap_sentences: int = 1

    @field_validator("target_tokens")
    @classmethod
    def _positive_tokens(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("target_tokens must be positive")
        return v


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"


class RetrievalConfig(BaseModel):
    strategy: Literal["dense", "hybrid", "hybrid_rerank"] = "dense"
    top_k: int = 5
    rerank_candidates: int = 20
    reranker: str | None = None


class LLMConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"


class EvalConfig(BaseModel):
    enabled: bool = True
    qa_pairs_path: str = "./eval/qa_pairs.jsonl"
    judge: str = "claude-sonnet-4-20250514"


class BenchmarkConfig(BaseModel):
    embedding_models: list[str] = []
    llms: list[str] = []
    rerankers: list[str] = []
    budget_limit_usd: float = 50.0


class SweepConfig(BaseModel):
    """Parameter grid for automated sweep evaluation."""

    chunking_strategies: list[Literal["fixed", "sentence", "semantic"]] = [
        "fixed", "sentence", "semantic",
    ]
    target_tokens: list[int] = [200, 400]
    retrieval_strategies: list[Literal["dense", "hybrid", "hybrid_rerank"]] = [
        "dense", "hybrid_rerank",
    ]
    top_k: list[int] = [5]

    @field_validator("target_tokens")
    @classmethod
    def _all_positive(cls, v: list[int]) -> list[int]:
        for t in v:
            if t <= 0:
                raise ValueError("All target_tokens values must be positive")
        return v


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8001
    metadata_filters: list[str] = []


class FluxRAGConfig(BaseModel):
    """Root configuration — validates and loads a complete domain.yaml."""

    domain: DomainConfig
    corpus: CorpusConfig
    chunking: ChunkingConfig = ChunkingConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    llm: LLMConfig = LLMConfig()
    eval: EvalConfig = EvalConfig()
    benchmark: BenchmarkConfig = BenchmarkConfig()
    sweep: SweepConfig = SweepConfig()
    api: APIConfig = APIConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> FluxRAGConfig:
        """Load and validate a domain.yaml configuration file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path) as f:
            raw = yaml.safe_load(f)
        return cls.model_validate(raw)
