"""Abstract vector store interface. ChromaDB and Qdrant backends implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from fluxrag.core.schema import EmbeddedChunk, RetrievalResult


class AbstractVectorStore(ABC):
    """Base interface for vector store backends."""

    @abstractmethod
    def add(self, chunks: list[EmbeddedChunk]) -> None:
        """Add embedded chunks to the store."""
        ...

    @abstractmethod
    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        """Search for similar chunks by embedding vector."""
        ...

    @abstractmethod
    def count(self) -> int:
        """Return total number of stored chunks."""
        ...

    @abstractmethod
    def reset(self) -> None:
        """Delete all stored data."""
        ...
