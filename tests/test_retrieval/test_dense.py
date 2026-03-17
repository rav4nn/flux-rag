"""Tests for dense retrieval strategy."""

import pytest

from fluxrag.core.schema import Chunk, EmbeddedChunk
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.embedding.local import LocalEmbedder
from fluxrag.retrieval.dense import DenseRetriever


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def retriever(embedder: LocalEmbedder) -> DenseRetriever:
    store = ChromaDBStore(collection_name="test_retrieval")
    store.reset()

    texts = [
        "Espresso requires a fine grind and 9 bars of pressure.",
        "Pour over coffee uses a medium grind and gravity for extraction.",
        "French press uses a coarse grind and steeps for 4 minutes.",
        "The ideal water temperature for coffee is between 90-96 degrees Celsius.",
        "Python is a popular programming language for data science.",
    ]

    chunks = []
    for i, text in enumerate(texts):
        chunk = Chunk(text=text, document_id="doc1", index=i, metadata={"source_type": "txt"})
        embedding = embedder.embed([text])[0]
        chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    store.add(chunks)
    return DenseRetriever(embedder, store)


def test_retrieves_relevant_results(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("What grind size for espresso?", top_k=3)
    assert len(results) == 3
    # Espresso chunk should be most relevant
    assert "espresso" in results[0].chunk.text.lower() or "grind" in results[0].chunk.text.lower()


def test_unrelated_query_still_returns(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("quantum physics", top_k=2)
    assert len(results) == 2


def test_top_k_limits_results(retriever: DenseRetriever) -> None:
    results = retriever.retrieve("coffee", top_k=2)
    assert len(results) == 2


def test_strategy_name(retriever: DenseRetriever) -> None:
    assert retriever.strategy_name == "dense"
