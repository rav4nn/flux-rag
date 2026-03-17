"""Abstract chunker interface. All chunking strategies implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fluxrag.core.schema import Chunk, Document


class AbstractChunker(ABC):
    """Base interface for all chunking strategies.

    Strategies: fixed-size, sentence-boundary, semantic similarity.
    """

    @abstractmethod
    def chunk(self, document: Document, target_tokens: int = 200, **kwargs: object) -> list[Chunk]:
        """Split a document into chunks.

        Args:
            document: The document to chunk.
            target_tokens: Target number of tokens per chunk.
            **kwargs: Strategy-specific options (e.g., overlap_sentences, similarity_threshold).

        Returns:
            Ordered list of Chunk objects.
        """
        ...

    @property
    @abstractmethod
    def strategy_name(self) -> str:
        """Return the strategy identifier (e.g., 'fixed', 'sentence', 'semantic')."""
        ...
