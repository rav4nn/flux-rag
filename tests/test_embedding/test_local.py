"""Tests for local embedding model."""

import pytest

from fluxrag.embedding.local import LocalEmbedder


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    """Load model once for all tests in this module."""
    return LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


def test_embed_returns_correct_dimensions(embedder: LocalEmbedder) -> None:
    vectors = embedder.embed(["test sentence"])
    assert len(vectors) == 1
    assert len(vectors[0]) == embedder.dimensions
    assert embedder.dimensions == 384


def test_embed_batch(embedder: LocalEmbedder) -> None:
    texts = ["coffee is great", "espresso extraction", "grind size matters"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert all(len(v) == 384 for v in vectors)


def test_embed_query(embedder: LocalEmbedder) -> None:
    vec = embedder.embed_query("what is the ideal brew temperature?")
    assert len(vec) == 384


def test_similar_texts_have_higher_cosine(embedder: LocalEmbedder) -> None:
    """Sanity check: semantically similar texts should be closer."""
    import math

    def cosine(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        return dot / (norm_a * norm_b) if norm_a and norm_b else 0.0

    v_coffee = embedder.embed_query("espresso extraction rate")
    v_similar = embedder.embed_query("coffee brewing extraction yield")
    v_unrelated = embedder.embed_query("quantum physics equations")

    sim_related = cosine(v_coffee, v_similar)
    sim_unrelated = cosine(v_coffee, v_unrelated)
    assert sim_related > sim_unrelated


def test_model_name(embedder: LocalEmbedder) -> None:
    assert "MiniLM" in embedder.model_name or "all-MiniLM" in embedder.model_name
