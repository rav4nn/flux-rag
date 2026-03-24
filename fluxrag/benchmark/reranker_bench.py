"""Matrix C benchmark runner — reranker model comparison."""

from __future__ import annotations

import logging
import time
from typing import Any

from fluxrag.benchmark.cost_tracker import CostTracker
from fluxrag.chunking.base import AbstractChunker
from fluxrag.core.schema import BenchmarkResult, Chunk, Document, EmbeddedChunk
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.eval.generate_qa import load_qa_pairs
from fluxrag.eval.harness import EvalHarness
from fluxrag.llm.base import AbstractLLM
from fluxrag.reranking.base import AbstractReranker
from fluxrag.reranking.factory import RERANKER_MODELS, create_reranker
from fluxrag.retrieval.hybrid import HybridRetriever
from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever

logger = logging.getLogger(__name__)


class MatrixCRunner:
    """Runs Matrix C: Reranker benchmark.

    Fixed chunking + retrieval + embedding (Matrix A + B winners), fixed LLM.
    Variable: reranker model (up to 5 configs).
    """

    def __init__(
        self,
        documents: list[Document],
        chunker: AbstractChunker,
        embedder: AbstractEmbedder,
        generator: AbstractLLM,
        judge: AbstractLLM,
        qa_pairs_path: str,
        reranker_models: list[str] | None = None,
        cost_tracker: CostTracker | None = None,
        target_tokens: int = 200,
        top_k: int = 5,
        rerank_candidates: int = 20,
    ) -> None:
        self.documents = documents
        self.chunker = chunker
        self.embedder = embedder
        self.generator = generator
        self.judge = judge
        self.qa_pairs = load_qa_pairs(qa_pairs_path)
        self.reranker_models = reranker_models or list(RERANKER_MODELS.keys())
        self.cost_tracker = cost_tracker or CostTracker()
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def run(self) -> BenchmarkResult:
        """Run all reranker configurations."""
        # Chunk and embed once — both are fixed
        chunks = self._chunk_all()
        logger.info("[Matrix C] %d chunks from %d documents", len(chunks), len(self.documents))

        store = self._build_store(chunks)
        results: list[dict[str, Any]] = []

        for model_name in self.reranker_models:
            self.cost_tracker.check_budget()
            info = RERANKER_MODELS.get(model_name, {})

            logger.info("[Matrix C] Evaluating reranker: %s", model_name)
            start = time.time()

            try:
                reranker = create_reranker(model_name, cost_tracker=self.cost_tracker)

                # Always use hybrid + rerank strategy for reranker comparison
                hybrid = HybridRetriever(self.embedder, store, chunks)
                retriever = HybridRerankRetriever(
                    hybrid, reranker, candidates=self.rerank_candidates,
                )

                harness = EvalHarness(
                    retriever=retriever,
                    generator=self.generator,
                    judge=self.judge,
                    top_k=self.top_k,
                )
                eval_result = harness.evaluate(self.qa_pairs)
                elapsed = time.time() - start

                results.append({
                    "model": model_name,
                    "type": "Local" if info.get("type") == "local" else "API",
                    "context_recall": eval_result.context_recall,
                    "context_precision": eval_result.context_precision,
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "cost_per_search": self._get_cost_per_search(model_name),
                    "error": None,
                })

            except Exception as e:
                elapsed = time.time() - start
                logger.error("[Matrix C] Failed for %s: %s", model_name, e)
                results.append({
                    "model": model_name,
                    "type": "Local" if info.get("type") == "local" else "API",
                    "context_recall": 0.0,
                    "context_precision": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "cost_per_search": self._get_cost_per_search(model_name),
                    "error": str(e),
                })

        store.reset()
        return BenchmarkResult(matrix_name="C", rows=results)

    def _chunk_all(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(self.chunker.chunk(doc, target_tokens=self.target_tokens))
        return chunks

    def _build_store(self, chunks: list[Chunk]) -> ChromaDBStore:
        store = ChromaDBStore(collection_name="matrix_c")
        store.reset()

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e)
                for c, e in zip(batch, embeddings)
            ]
            store.add(embedded)

        return store

    def _get_cost_per_search(self, model_name: str) -> str:
        info = RERANKER_MODELS.get(model_name, {})
        if info.get("type") == "local":
            return "$0"
        if info.get("type") == "cohere":
            return "$0.001"
        if info.get("type") == "jina":
            return "~$0.001"
        return "N/A"
