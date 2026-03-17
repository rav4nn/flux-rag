"""Strategy C: Semantic similarity chunking."""

from __future__ import annotations

import re

from fluxrag.chunking.base import AbstractChunker
from fluxrag.core.schema import Chunk, Document
from fluxrag.embedding.base import AbstractEmbedder

_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"])'
    r'|(?<=[.!?])\s*\n'
)


def _split_sentences(text: str) -> list[str]:
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticChunker(AbstractChunker):
    """Group consecutive sentences while semantic similarity stays above threshold.

    Split on topic shift (similarity drop below threshold).
    Variable chunk length, respects semantic boundaries.
    """

    DEFAULT_THRESHOLD = 0.7
    MIN_CHUNK_SENTENCES = 2

    def __init__(self, embedder: AbstractEmbedder) -> None:
        self._embedder = embedder

    def chunk(self, document: Document, target_tokens: int = 200, **kwargs: object) -> list[Chunk]:
        threshold = float(kwargs.get("similarity_threshold", self.DEFAULT_THRESHOLD))
        sentences = _split_sentences(document.text)

        if not sentences:
            return []

        if len(sentences) <= self.MIN_CHUNK_SENTENCES:
            return [
                Chunk(
                    text=document.text.strip(),
                    document_id=document.id,
                    index=0,
                    metadata={**document.metadata, "chunking_strategy": "semantic"},
                )
            ]

        # Embed all sentences in one batch
        embeddings = self._embedder.embed(sentences)

        # Group by similarity
        groups: list[list[int]] = []
        current_group: list[int] = [0]

        for i in range(1, len(sentences)):
            sim = _cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim >= threshold and len(current_group) < 20:
                current_group.append(i)
            else:
                groups.append(current_group)
                current_group = [i]

        # Flush last group
        if current_group:
            groups.append(current_group)

        # Merge tiny trailing group into previous if needed
        if len(groups) > 1 and len(groups[-1]) < self.MIN_CHUNK_SENTENCES:
            groups[-2].extend(groups[-1])
            groups.pop()

        # Build chunks
        chunks: list[Chunk] = []
        for idx, group in enumerate(groups):
            chunk_text = " ".join(sentences[i] for i in group)
            chunks.append(
                Chunk(
                    text=chunk_text,
                    document_id=document.id,
                    index=idx,
                    metadata={**document.metadata, "chunking_strategy": "semantic"},
                )
            )

        return chunks

    @property
    def strategy_name(self) -> str:
        return "semantic"
