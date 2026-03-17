"""Abstract reranker interface. All reranker models implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fluxrag.core.schema import RerankedResult, RetrievalResult


class AbstractReranker(ABC):
    """Base interface for all reranker models.

    Covers local cross-encoders and API rerankers (Cohere, Jina).
    """

    @abstractmethod
    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        """Rerank retrieval results using a cross-encoder or reranking API.

        Args:
            query: The original query.
            results: Candidate retrieval results to rerank.
            top_k: Number of results to return after reranking.

        Returns:
            Reranked results ordered by reranker score (highest first).
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the reranker model identifier."""
        ...
