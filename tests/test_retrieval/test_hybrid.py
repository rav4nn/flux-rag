"""Tests for hybrid retrieval (BM25 + dense + RRF)."""

import pytest

from fluxrag.core.schema import Chunk, EmbeddedChunk
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.embedding.local import LocalEmbedder
from fluxrag.retrieval.hybrid import HybridRetriever


COFFEE_CHUNKS = [
    "The Baratza Encore is a popular entry-level burr grinder for home brewing.",
    "Espresso requires a fine grind and 9 bars of pressure for proper extraction.",
    "Pour over coffee uses a medium grind and gravity-based extraction.",
    "French press uses a coarse grind and steeps for 4 minutes in hot water.",
    "The ideal water temperature for coffee is between 90-96 degrees Celsius.",
]


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def retriever(embedder: LocalEmbedder) -> HybridRetriever:
    store = ChromaDBStore(collection_name="test_hybrid")
    store.reset()

    chunks: list[Chunk] = []
    embedded: list[EmbeddedChunk] = []
    for i, text in enumerate(COFFEE_CHUNKS):
        chunk = Chunk(text=text, document_id="doc1", index=i, metadata={"source_type": "txt"})
        chunks.append(chunk)
        embedding = embedder.embed([text])[0]
        embedded.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    store.add(embedded)
    return HybridRetriever(embedder, store, chunks)


def test_hybrid_retrieves_relevant(retriever: HybridRetriever) -> None:
    results = retriever.retrieve("espresso grind", top_k=3)
    assert len(results) == 3
    texts = " ".join(r.chunk.text.lower() for r in results)
    assert "espresso" in texts


def test_hybrid_catches_exact_keyword(retriever: HybridRetriever) -> None:
    """BM25 should boost exact matches that dense might miss."""
    results = retriever.retrieve("Baratza Encore", top_k=3)
    texts = [r.chunk.text for r in results]
    assert any("Baratza" in t for t in texts)


def test_hybrid_strategy_name(retriever: HybridRetriever) -> None:
    assert retriever.strategy_name == "hybrid"


def test_rrf_scores_are_ordered(retriever: HybridRetriever) -> None:
    results = retriever.retrieve("coffee brewing", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
