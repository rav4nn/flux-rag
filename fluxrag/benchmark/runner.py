"""Benchmark orchestrator — runs matrices sequentially."""

from __future__ import annotations

import logging
import time
from typing import Any

from fluxrag.benchmark.cost_tracker import CostTracker
from fluxrag.chunking.base import AbstractChunker
from fluxrag.chunking.fixed import FixedChunker
from fluxrag.chunking.sentence import SentenceChunker
from fluxrag.chunking.semantic import SemanticChunker
from fluxrag.core.schema import BenchmarkResult, Chunk, Document, EmbeddedChunk, EvalResult
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.eval.harness import EvalHarness
from fluxrag.eval.generate_qa import load_qa_pairs
from fluxrag.llm.base import AbstractLLM
from fluxrag.reranking.cross_encoder import CrossEncoderReranker
from fluxrag.retrieval.base import AbstractRetriever
from fluxrag.retrieval.dense import DenseRetriever
from fluxrag.retrieval.hybrid import HybridRetriever
from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever

logger = logging.getLogger(__name__)


CHUNKING_LABELS = {"fixed": "A — Fixed", "sentence": "B — Sentence", "semantic": "C — Semantic"}
RETRIEVAL_LABELS = {"dense": "1 — Dense", "hybrid": "2 — Hybrid", "hybrid_rerank": "3 — Hybrid + Rerank"}


class MatrixARunner:
    """Runs Matrix A: 3×3 Chunking × Retrieval benchmark.

    Fixed embedding (all-MiniLM-L6-v2), fixed reranker (ms-marco-MiniLM),
    fixed LLM (Claude Sonnet 4).
    """

    def __init__(
        self,
        documents: list[Document],
        embedder: AbstractEmbedder,
        generator: AbstractLLM,
        judge: AbstractLLM,
        qa_pairs_path: str,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        target_tokens: int = 200,
        top_k: int = 5,
        rerank_candidates: int = 20,
    ) -> None:
        self.documents = documents
        self.embedder = embedder
        self.generator = generator
        self.judge = judge
        self.qa_pairs = load_qa_pairs(qa_pairs_path)
        self.reranker_model = reranker_model
        self.target_tokens = target_tokens
        self.top_k = top_k
        self.rerank_candidates = rerank_candidates

    def run(self) -> BenchmarkResult:
        """Run all 9 configurations of Matrix A."""
        chunking_strategies = ["fixed", "sentence", "semantic"]
        retrieval_strategies = ["dense", "hybrid", "hybrid_rerank"]

        results: list[dict[str, Any]] = []
        reranker = None  # Lazy init only if needed

        for cs in chunking_strategies:
            # Chunk documents
            chunker = self._create_chunker(cs)
            chunks = self._chunk_all(chunker)
            logger.info("[Matrix A] %s chunking: %d chunks", cs, len(chunks))

            # Embed and store
            store = self._build_store(chunks, collection_name=f"matrix_a_{cs}")

            for rs in retrieval_strategies:
                config_name = f"{cs[0].upper()}{retrieval_strategies.index(rs) + 1}"
                logger.info("[Matrix A] Running config %s: %s × %s", config_name, cs, rs)

                start = time.time()

                retriever = self._create_retriever(rs, chunks, store, reranker)

                # If hybrid_rerank and reranker not loaded yet, load it
                if rs == "hybrid_rerank" and reranker is None:
                    reranker = CrossEncoderReranker(self.reranker_model)
                    retriever = self._create_retriever(rs, chunks, store, reranker)

                harness = EvalHarness(
                    retriever=retriever,
                    generator=self.generator,
                    judge=self.judge,
                    top_k=self.top_k,
                )
                eval_result = harness.evaluate(self.qa_pairs)
                elapsed = time.time() - start

                results.append({
                    "config": config_name,
                    "chunking": CHUNKING_LABELS[cs],
                    "retrieval": RETRIEVAL_LABELS[rs],
                    "context_recall": eval_result.context_recall,
                    "context_precision": eval_result.context_precision,
                    "faithfulness": eval_result.faithfulness,
                    "answer_relevancy": eval_result.answer_relevancy,
                    "latency_ms": round(elapsed / len(self.qa_pairs) * 1000) if self.qa_pairs else 0,
                    "num_chunks": len(chunks),
                })

            # Clean up store to free memory
            store.reset()

        return BenchmarkResult(matrix_name="A", rows=results)

    def _create_chunker(self, strategy: str) -> AbstractChunker:
        if strategy == "fixed":
            return FixedChunker()
        elif strategy == "sentence":
            return SentenceChunker()
        elif strategy == "semantic":
            return SemanticChunker(self.embedder)
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_all(self, chunker: AbstractChunker) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(chunker.chunk(doc, target_tokens=self.target_tokens))
        return chunks

    def _build_store(self, chunks: list[Chunk], collection_name: str) -> ChromaDBStore:
        store = ChromaDBStore(collection_name=collection_name)
        store.reset()

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(batch, embeddings)
            ]
            store.add(embedded)

        return store

    def _create_retriever(
        self,
        strategy: str,
        chunks: list[Chunk],
        store: ChromaDBStore,
        reranker: CrossEncoderReranker | None,
    ) -> AbstractRetriever:
        if strategy == "dense":
            return DenseRetriever(self.embedder, store)
        elif strategy == "hybrid":
            return HybridRetriever(self.embedder, store, chunks)
        elif strategy == "hybrid_rerank":
            hybrid = HybridRetriever(self.embedder, store, chunks)
            if reranker is None:
                reranker = CrossEncoderReranker(self.reranker_model)
            return HybridRerankRetriever(hybrid, reranker, self.rerank_candidates)
        raise ValueError(f"Unknown retrieval strategy: {strategy}")
