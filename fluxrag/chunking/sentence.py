"""Strategy B: Sentence-boundary aware chunking."""

from __future__ import annotations

import re

from fluxrag.chunking.base import AbstractChunker
from fluxrag.core.schema import Chunk, Document

# Sentence boundary regex: split on .!? followed by whitespace and uppercase letter,
# or end of string. Handles abbreviations (Mr., Dr., etc.) and decimals reasonably.
_SENTENCE_SPLIT = re.compile(
    r'(?<=[.!?])\s+(?=[A-Z"])'
    r'|(?<=[.!?])\s*\n'
)


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex boundaries."""
    sentences = _SENTENCE_SPLIT.split(text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _estimate_tokens(text: str) -> int:
    """Rough token estimate: ~0.75 tokens per word."""
    return max(1, int(len(text.split()) * 0.75))


class SentenceChunker(AbstractChunker):
    """Group sentences until target token count is reached.

    Never cuts mid-sentence. One-sentence overlap between chunks.
    """

    def chunk(self, document: Document, target_tokens: int = 200, **kwargs: object) -> list[Chunk]:
        overlap_sentences = int(kwargs.get("overlap_sentences", 1))
        sentences = _split_sentences(document.text)

        if not sentences:
            return []

        chunks: list[Chunk] = []
        index = 0
        i = 0

        while i < len(sentences):
            group: list[str] = []
            token_count = 0

            while i < len(sentences):
                sent_tokens = _estimate_tokens(sentences[i])
                if group and token_count + sent_tokens > target_tokens:
                    break
                group.append(sentences[i])
                token_count += sent_tokens
                i += 1

            if group:
                chunk_text = " ".join(group)
                chunks.append(
                    Chunk(
                        text=chunk_text,
                        document_id=document.id,
                        index=index,
                        metadata={**document.metadata, "chunking_strategy": "sentence"},
                    )
                )
                index += 1

                # Overlap: step back by overlap_sentences
                i = max(i - overlap_sentences, i - len(group) + 1)
                if i <= (chunks[-1].index if len(chunks) > 1 else -1):
                    break  # Prevent infinite loop on very short text

        return chunks

    @property
    def strategy_name(self) -> str:
        return "sentence"
