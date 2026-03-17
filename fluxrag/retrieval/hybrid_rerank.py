"""Strategy 3: Hybrid search + Cross-encoder reranking."""

from __future__ import annotations

from typing import Any

from fluxrag.core.schema import RetrievalResult
from fluxrag.reranking.base import AbstractReranker
from fluxrag.retrieval.base import AbstractRetriever
from fluxrag.retrieval.hybrid import HybridRetriever


class HybridRerankRetriever(AbstractRetriever):
    """Hybrid search (BM25 + dense) with cross-encoder reranking.

    Retrieves top-N candidates via hybrid search, then reranks
    with a cross-encoder to maximize precision.
    """

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        reranker: AbstractReranker,
        rerank_candidates: int = 20,
    ) -> None:
        self._hybrid = hybrid_retriever
        self._reranker = reranker
        self._rerank_candidates = rerank_candidates

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        # Get more candidates than needed for reranking
        candidates = self._hybrid.retrieve(
            query, top_k=self._rerank_candidates, filters=filters
        )

        if not candidates:
            return []

        # Rerank
        reranked = self._reranker.rerank(query, candidates, top_k=top_k)

        # Convert RerankedResult back to RetrievalResult
        return [
            RetrievalResult(chunk=r.chunk, score=r.score)
            for r in reranked
        ]

    @property
    def strategy_name(self) -> str:
        return "hybrid_rerank"
