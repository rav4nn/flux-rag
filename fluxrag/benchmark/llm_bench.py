"""Matrix D benchmark runner — LLM generation model comparison."""

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
from fluxrag.llm.factory import LLM_MODELS, create_llm, get_cost_per_m
from fluxrag.reranking.base import AbstractReranker
from fluxrag.retrieval.base import AbstractRetriever
from fluxrag.retrieval.dense import DenseRetriever
from fluxrag.retrieval.hybrid import HybridRetriever
from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever

logger = logging.getLogger(__name__)


class MatrixDRunner:
    """Runs Matrix D: LLM Generation benchmark.

    Fixed chunking + retrieval + embedding + reranker (winners from A + B + C).
    Variable: LLM generation model (up to 8 configs).
    Judge is always Claude Sonnet 4 (fixed).
    """

    def __init__(
        self,
        documents: list[Document],
        chunker: AbstractChunker,
        embedder: AbstractEmbedder,
        retrieval_strategy: str,
        reranker: AbstractReranker | None,
        judge: AbstractLLM,
        qa_pairs_path: str,
        llm_models: list[str] | None = None,
        cost_tracker: CostTracker | None = None,
        target_tokens: int = 200,
        top_k: int = 5,
        rerank_candidates: int = 20,
    ) -> None:
        self.documents = documents
        self.chunker = chunker
        self.embedder = embedder
        self.retrieval_strategy = retrieval_strategy
        self.reranker = reranker
        self.judge = judge
        self.qa_pairs = load_qa_pairs(qa_pairs_path)
        self.llm_models = llm_models or list(LLM_MODELS.keys())
        self.cost_tracker = cost_tracker or CostTracker()
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def run(self) -> BenchmarkResult:
        """Run all LLM configurations."""
        # Chunk, embed, and store once — all retrieval components are fixed
        chunks = self._chunk_all()
        logger.info("[Matrix D] %d chunks from %d documents", len(chunks), len(self.documents))

        store = self._build_store(chunks)
        retriever = self._create_retriever(chunks, store)
        results: list[dict[str, Any]] = []

        for model_name in self.llm_models:
            self.cost_tracker.check_budget()
            info = LLM_MODELS.get(model_name, {})
            input_cost, output_cost = get_cost_per_m(model_name)

            logger.info("[Matrix D] Evaluating LLM: %s", model_name)
            start = time.time()

            try:
                generator = create_llm(model_name)

                harness = EvalHarness(
                    retriever=retriever,
                    generator=generator,
                    judge=self.judge,
                    top_k=self.top_k,
                )
                eval_result = harness.evaluate(self.qa_pairs)
                elapsed = time.time() - start

                results.append({
                    "model": model_name,
                    "provider": info.get("type", "unknown"),
                    "context_recall": eval_result.context_recall,
                    "context_precision": eval_result.context_precision,
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "input_cost_per_m": f"${input_cost:.2f}",
                    "output_cost_per_m": f"${output_cost:.2f}",
                    "error": None,
                })

            except Exception as e:
                elapsed = time.time() - start
                logger.error("[Matrix D] Failed for %s: %s", model_name, e)
                results.append({
                    "model": model_name,
                    "provider": info.get("type", "unknown"),
                    "context_recall": 0.0,
                    "context_precision": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "input_cost_per_m": f"${input_cost:.2f}",
                    "output_cost_per_m": f"${output_cost:.2f}",
                    "error": str(e),
                })

        store.reset()
        return BenchmarkResult(matrix_name="D", rows=results)

    def _chunk_all(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(self.chunker.chunk(doc, target_tokens=self.target_tokens))
        return chunks

    def _build_store(self, chunks: list[Chunk]) -> ChromaDBStore:
        store = ChromaDBStore(collection_name="matrix_d")
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

    def _create_retriever(
        self,
        chunks: list[Chunk],
        store: ChromaDBStore,
    ) -> AbstractRetriever:
        if self.retrieval_strategy == "dense":
            return DenseRetriever(self.embedder, store)
        elif self.retrieval_strategy == "hybrid":
            return HybridRetriever(self.embedder, store, chunks)
        elif self.retrieval_strategy == "hybrid_rerank":
            hybrid = HybridRetriever(self.embedder, store, chunks)
            return HybridRerankRetriever(hybrid, self.reranker, self.rerank_candidates)
        raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")
