"""Strategy 2: Hybrid retrieval (BM25 + Dense via Reciprocal Rank Fusion)."""

from __future__ import annotations

from typing import Any

from rank_bm25 import BM25Okapi

from fluxrag.core.schema import Chunk, RetrievalResult
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.store import AbstractVectorStore
from fluxrag.retrieval.base import AbstractRetriever


class HybridRetriever(AbstractRetriever):
    """BM25 keyword search + dense vector search merged via RRF.

    Catches exact-match queries that dense search misses (grinder names,
    specific ratios, product models).
    """

    RRF_K = 60  # Reciprocal Rank Fusion constant

    def __init__(
        self,
        embedder: AbstractEmbedder,
        store: AbstractVectorStore,
        chunks: list[Chunk],
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._chunks = chunks
        self._chunk_map = {c.id: c for c in chunks}

        # Build BM25 index
        tokenized = [c.text.lower().split() for c in chunks]
        self._bm25 = BM25Okapi(tokenized)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        # Dense search
        dense_results = self._dense_search(query, top_k=top_k * 2, filters=filters)

        # BM25 search
        bm25_results = self._bm25_search(query, top_k=top_k * 2)

        # Merge via Reciprocal Rank Fusion
        return self._rrf_merge(dense_results, bm25_results, top_k)

    def _dense_search(
        self, query: str, top_k: int, filters: dict[str, Any] | None
    ) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed_query(query)
        return self._store.search(query_embedding, top_k=top_k, filters=filters)

    def _bm25_search(self, query: str, top_k: int) -> list[RetrievalResult]:
        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)

        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[RetrievalResult] = []
        for idx, score in scored_indices:
            if score > 0 and idx < len(self._chunks):
                results.append(RetrievalResult(chunk=self._chunks[idx], score=float(score)))
        return results

    def _rrf_merge(
        self,
        dense: list[RetrievalResult],
        bm25: list[RetrievalResult],
        top_k: int,
    ) -> list[RetrievalResult]:
        """Reciprocal Rank Fusion: score = sum(1 / (k + rank)) across lists."""
        rrf_scores: dict[str, float] = {}
        chunk_lookup: dict[str, Chunk] = {}

        for rank, r in enumerate(dense):
            rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0) + 1.0 / (self.RRF_K + rank + 1)
            chunk_lookup[r.chunk.id] = r.chunk

        for rank, r in enumerate(bm25):
            rrf_scores[r.chunk.id] = rrf_scores.get(r.chunk.id, 0) + 1.0 / (self.RRF_K + rank + 1)
            chunk_lookup[r.chunk.id] = r.chunk

        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)[:top_k]

        return [
            RetrievalResult(chunk=chunk_lookup[cid], score=rrf_scores[cid])
            for cid in sorted_ids
        ]

    @property
    def strategy_name(self) -> str:
        return "hybrid"
