"""Abstract retriever interface. All retrieval strategies implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fluxrag.core.schema import RetrievalResult


class AbstractRetriever(ABC):
    """Base interface for all retrieval strategies.

    Strategies: dense, hybrid (BM25 + dense), hybrid + rerank.
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Retrieve the most relevant chunks for a query.

        Args:
            query: The user's question.
            top_k: Number of results to return.
            filters: Optional metadata filters.

        Returns:
            List of RetrievalResult ordered by relevance (highest first).
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy identifier (e.g., 'dense', 'hybrid', 'hybrid_rerank')."""
        ...
