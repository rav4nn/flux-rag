"""Strategy A: Fixed-size chunking with overlap."""

from __future__ import annotations

from fluxrag.chunking.base import AbstractChunker
from fluxrag.core.schema import Chunk, Document


class FixedChunker(AbstractChunker):
    """Split text by character count with overlap.

    Simple baseline chunker. No sentence boundary awareness.
    """

    DEFAULT_CHUNK_SIZE = 800
    DEFAULT_OVERLAP_RATIO = 0.1

    def chunk(self, document: Document, target_tokens: int = 200, **kwargs: object) -> list[Chunk]:
        chunk_size = int(kwargs.get("chunk_size", self.DEFAULT_CHUNK_SIZE))
        overlap_ratio = float(kwargs.get("overlap_ratio", self.DEFAULT_OVERLAP_RATIO))
        overlap = int(chunk_size * overlap_ratio)
        step = max(chunk_size - overlap, 1)

        text = document.text
        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        index = 0

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document.id,
                        index=index,
                        metadata={**document.metadata, "chunking_strategy": "fixed"},
                    )
                )
                index += 1

            if end >= len(text):
                break
            start += step

        return chunks

    @property
    def strategy_name(self) -> str:
        return "fixed"
