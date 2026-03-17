"""Tests for semantic similarity chunking strategy."""

import pytest

from fluxrag.chunking.semantic import SemanticChunker
from fluxrag.core.schema import Document
from fluxrag.embedding.local import LocalEmbedder


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture(scope="module")
def chunker(embedder: LocalEmbedder) -> SemanticChunker:
    return SemanticChunker(embedder)


def _make_doc(text: str) -> Document:
    return Document(text=text, metadata={"source_type": "txt", "source_path": "test.txt"})


def test_groups_similar_sentences(chunker: SemanticChunker) -> None:
    """Sentences about the same topic should be grouped together."""
    text = (
        "Espresso uses finely ground coffee. The grind should be like table salt. "
        "Pressure of 9 bars forces water through the puck. "
        "The Milky Way galaxy contains hundreds of billions of stars. "
        "Galaxies are gravitationally bound systems of stars and dark matter. "
        "The observable universe contains over two trillion galaxies."
    )
    doc = _make_doc(text)
    # Use a higher threshold to force splits between very different topics
    chunks = chunker.chunk(doc, target_tokens=200, similarity_threshold=0.8)
    assert len(chunks) >= 2


def test_short_text_single_chunk(chunker: SemanticChunker) -> None:
    doc = _make_doc("A short sentence.")
    chunks = chunker.chunk(doc, target_tokens=200)
    assert len(chunks) == 1


def test_homogeneous_text_fewer_chunks(chunker: SemanticChunker) -> None:
    """Text about one topic should produce fewer chunks than mixed topics."""
    coffee_text = (
        "Coffee extraction depends on grind size. Finer grinds extract faster. "
        "Water temperature also affects extraction rate. Higher temperatures extract more. "
        "Brew time is the third key variable. Longer brew times increase extraction."
    )
    doc = _make_doc(coffee_text)
    chunks = chunker.chunk(doc, target_tokens=200, similarity_threshold=0.5)
    # Homogeneous text should stay together
    assert len(chunks) <= 3


def test_strategy_name(chunker: SemanticChunker) -> None:
    assert chunker.strategy_name == "semantic"


def test_metadata_includes_strategy(chunker: SemanticChunker) -> None:
    doc = _make_doc("First topic sentence. Second topic sentence.")
    chunks = chunker.chunk(doc, target_tokens=200)
    assert all(c.metadata.get("chunking_strategy") == "semantic" for c in chunks)
