"""Tests for ChromaDB vector store."""

import pytest

from fluxrag.core.schema import Chunk, EmbeddedChunk
from fluxrag.embedding.chromadb_store import ChromaDBStore


@pytest.fixture
def store() -> ChromaDBStore:
    s = ChromaDBStore(collection_name="test_collection")
    s.reset()
    return s


def _make_embedded_chunk(text: str, doc_id: str, index: int, embedding: list[float]) -> EmbeddedChunk:
    chunk = Chunk(text=text, document_id=doc_id, index=index, metadata={"source_type": "txt"})
    return EmbeddedChunk(chunk=chunk, embedding=embedding)


def test_add_and_count(store: ChromaDBStore) -> None:
    chunks = [
        _make_embedded_chunk("coffee text", "doc1", 0, [1.0, 0.0, 0.0]),
        _make_embedded_chunk("tea text", "doc1", 1, [0.0, 1.0, 0.0]),
    ]
    store.add(chunks)
    assert store.count() == 2


def test_search_returns_results(store: ChromaDBStore) -> None:
    chunks = [
        _make_embedded_chunk("espresso extraction", "doc1", 0, [1.0, 0.0, 0.0]),
        _make_embedded_chunk("green tea leaves", "doc1", 1, [0.0, 1.0, 0.0]),
        _make_embedded_chunk("coffee grind size", "doc1", 2, [0.9, 0.1, 0.0]),
    ]
    store.add(chunks)

    results = store.search([1.0, 0.0, 0.0], top_k=2)
    assert len(results) == 2
    assert results[0].chunk.text == "espresso extraction"
    assert results[0].score > results[1].score


def test_search_empty_store(store: ChromaDBStore) -> None:
    results = store.search([1.0, 0.0, 0.0], top_k=5)
    assert results == []


def test_reset_clears_data(store: ChromaDBStore) -> None:
    chunks = [_make_embedded_chunk("text", "doc1", 0, [1.0, 0.0, 0.0])]
    store.add(chunks)
    assert store.count() == 1
    store.reset()
    assert store.count() == 0
