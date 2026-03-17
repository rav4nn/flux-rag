"""Pipeline orchestrator — ties all FluxRAG components together."""

from __future__ import annotations

from typing import Any

from fluxrag.core.config import FluxRAGConfig
from fluxrag.core.schema import BenchmarkResult, Document, EvalResult, QueryResult


class Pipeline:
    """Main orchestrator for the FluxRAG pipeline.

    Usage:
        pipeline = Pipeline.from_config("domain.yaml")
        pipeline.ingest()
        pipeline.build()
        results = pipeline.evaluate()
    """

    def __init__(self, config: FluxRAGConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config_path: str) -> Pipeline:
        """Load a pipeline from a domain.yaml config file."""
        config = FluxRAGConfig.from_yaml(config_path)
        return cls(config)

    def ingest(self) -> list[Document]:
        """Parse all configured sources into unified Document objects."""
        raise NotImplementedError("Ingestion not yet implemented (Phase 1)")

    def build(self) -> None:
        """Chunk, embed, and store all documents in the vector store."""
        raise NotImplementedError("Build not yet implemented (Phase 2)")

    def evaluate(self) -> EvalResult:
        """Run RAGAS evaluation against the QA test set."""
        raise NotImplementedError("Evaluation not yet implemented (Phase 2)")

    def benchmark(self) -> BenchmarkResult:
        """Run all configured benchmark matrices."""
        raise NotImplementedError("Benchmarking not yet implemented (Phase 3+)")

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> QueryResult:
        """Query the knowledge base."""
        raise NotImplementedError("Query not yet implemented (Phase 2)")

    def serve(self) -> None:
        """Start the FastAPI server."""
        from fluxrag.api.server import run_server

        run_server(
            config_path="",
            host=self.config.api.host,
            port=self.config.api.port,
        )
