"""Abstract embedder interface. All embedding models implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractEmbedder(ABC):
    """Base interface for all embedding models.

    Covers local (sentence-transformers) and API (OpenAI, Cohere, Voyage) models.
    """

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each a list of floats).
        """
        ...

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string.

        Some models use different prefixes for queries vs documents.

        Args:
            query: The query text.

        Returns:
            Embedding vector.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the embedding dimensionality."""
        ...
