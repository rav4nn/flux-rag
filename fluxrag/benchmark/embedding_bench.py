"""Matrix B benchmark runner — embedding model comparison."""

from __future__ import annotations

import logging
import time
from typing import Any

from fluxrag.benchmark.cost_tracker import CostTracker
from fluxrag.chunking.base import AbstractChunker
from fluxrag.core.schema import BenchmarkResult, Chunk, Document, EmbeddedChunk
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.embedding.factory import EMBEDDING_MODELS, create_embedder
from fluxrag.eval.generate_qa import load_qa_pairs
from fluxrag.eval.harness import EvalHarness
from fluxrag.llm.base import AbstractLLM
from fluxrag.reranking.cross_encoder import CrossEncoderReranker
from fluxrag.retrieval.base import AbstractRetriever
from fluxrag.retrieval.dense import DenseRetriever
from fluxrag.retrieval.hybrid import HybridRetriever
from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever

logger = logging.getLogger(__name__)


class MatrixBRunner:
    """Runs Matrix B: Embedding Model benchmark.

    Fixed chunking + retrieval (Matrix A winner), fixed reranker, fixed LLM.
    Variable: embedding model (up to 8 configs).
    """

    def __init__(
        self,
        documents: list[Document],
        chunker: AbstractChunker,
        retrieval_strategy: str,
        generator: AbstractLLM,
        judge: AbstractLLM,
        qa_pairs_path: str,
        embedding_models: list[str] | None = None,
        cost_tracker: CostTracker | None = None,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        target_tokens: int = 200,
        top_k: int = 5,
        rerank_candidates: int = 20,
    ) -> None:
        self.documents = documents
        self.chunker = chunker
        self.retrieval_strategy = retrieval_strategy
        self.generator = generator
        self.judge = judge
        self.qa_pairs = load_qa_pairs(qa_pairs_path)
        self.embedding_models = embedding_models or list(EMBEDDING_MODELS.keys())
        self.cost_tracker = cost_tracker or CostTracker()
        self.reranker_model = reranker_model
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def run(self) -> BenchmarkResult:
        """Run all embedding model configurations."""
        # Chunk once — chunking strategy is fixed
        chunks = self._chunk_all()
        logger.info("[Matrix B] %d chunks from %d documents", len(chunks), len(self.documents))

        reranker = None
        results: list[dict[str, Any]] = []

        for model_name in self.embedding_models:
            self.cost_tracker.check_budget()
            info = EMBEDDING_MODELS.get(model_name, {})

            logger.info("[Matrix B] Evaluating: %s", model_name)
            start = time.time()

            try:
                embedder = create_embedder(model_name, cost_tracker=self.cost_tracker)

                # Build store with this embedder
                collection = f"matrix_b_{model_name.replace('/', '_')}"
                store = self._build_store(chunks, embedder, collection_name=collection)

                # Create retriever with fixed strategy
                if self.retrieval_strategy == "hybrid_rerank" and reranker is None:
                    reranker = CrossEncoderReranker(self.reranker_model)
                retriever = self._create_retriever(embedder, chunks, store, reranker)

                # Evaluate
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
                    "dimensions": info.get("dimensions", embedder.dimensions),
                    "context_recall": eval_result.context_recall,
                    "context_precision": eval_result.context_precision,
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "cost_per_m_tokens": self._get_cost_per_m(model_name),
                    "error": None,
                })

                store.reset()

            except Exception as e:
                elapsed = time.time() - start
                logger.error("[Matrix B] Failed for %s: %s", model_name, e)
                results.append({
                    "model": model_name,
                    "type": "Local" if info.get("type") == "local" else "API",
                    "dimensions": info.get("dimensions", 0),
                    "context_recall": 0.0,
                    "context_precision": 0.0,
                    "faithfulness": 0.0,
                    "answer_relevancy": 0.0,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "cost_per_m_tokens": self._get_cost_per_m(model_name),
                    "error": str(e),
                })

        return BenchmarkResult(matrix_name="B", rows=results)

    def _chunk_all(self) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(self.chunker.chunk(doc, target_tokens=self.target_tokens))
        return chunks

    def _build_store(
        self,
        chunks: list[Chunk],
        embedder: AbstractEmbedder,
        collection_name: str,
    ) -> ChromaDBStore:
        store = ChromaDBStore(collection_name=collection_name)
        store.reset()

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = embedder.embed([c.text for c in batch])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e)
                for c, e in zip(batch, embeddings)
            ]
            store.add(embedded)

        return store

    def _create_retriever(
        self,
        embedder: AbstractEmbedder,
        chunks: list[Chunk],
        store: ChromaDBStore,
        reranker: CrossEncoderReranker | None,
    ) -> AbstractRetriever:
        if self.retrieval_strategy == "dense":
            return DenseRetriever(embedder, store)
        elif self.retrieval_strategy == "hybrid":
            return HybridRetriever(embedder, store, chunks)
        elif self.retrieval_strategy == "hybrid_rerank":
            hybrid = HybridRetriever(embedder, store, chunks)
            if reranker is None:
                reranker = CrossEncoderReranker(self.reranker_model)
            return HybridRerankRetriever(hybrid, reranker, self.rerank_candidates)
        raise ValueError(f"Unknown retrieval strategy: {self.retrieval_strategy}")

    def _get_cost_per_m(self, model_name: str) -> str:
        """Return display cost string for the model."""
        from fluxrag.embedding.openai import OPENAI_PRICING
        from fluxrag.embedding.cohere import COHERE_PRICING
        from fluxrag.embedding.voyage import VOYAGE_PRICING

        info = EMBEDDING_MODELS.get(model_name, {})
        if info.get("type") == "local":
            return "$0"

        bare = model_name.split("/", 1)[1] if "/" in model_name else model_name
        price = (
            OPENAI_PRICING.get(bare)
            or COHERE_PRICING.get(bare)
            or VOYAGE_PRICING.get(bare)
        )
        return f"${price:.2f}" if price else "N/A"
