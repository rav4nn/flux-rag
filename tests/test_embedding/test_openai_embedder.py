"""Tests for OpenAI embedding client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.benchmark.cost_tracker import CostTracker


def _make_openai_mock():
    """Create a mock openai module with a working OpenAI client."""
    mock_module = MagicMock()

    def make_response(model, input):
        items = []
        for _ in input:
            item = MagicMock()
            item.embedding = [0.1] * 1536
            items.append(item)
        resp = MagicMock()
        resp.data = items
        return resp

    mock_module.OpenAI.return_value.embeddings.create.side_effect = make_response
    return mock_module


@pytest.fixture(autouse=True)
def mock_openai():
    """Inject mock openai module before any import."""
    mock_mod = _make_openai_mock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        # Clear cached import so our module re-imports
        if "fluxrag.embedding.openai" in sys.modules:
            del sys.modules["fluxrag.embedding.openai"]
        yield mock_mod


def test_embed_returns_correct_dimensions() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vectors = embedder.embed(["test sentence"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1536


def test_embed_batch() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    texts = ["coffee is great", "espresso extraction", "grind size matters"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3
    assert all(len(v) == 1536 for v in vectors)


def test_embed_query() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vec = embedder.embed_query("what is the ideal brew temperature?")
    assert len(vec) == 1536


def test_model_name() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    assert embedder.model_name == "openai/text-embedding-3-small"


def test_dimensions_large_model() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-large")
    assert embedder.dimensions == 3072


def test_cost_tracking() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    tracker = CostTracker()
    embedder = OpenAIEmbedder(model="text-embedding-3-small", cost_tracker=tracker)
    embedder.embed(["hello world"])
    assert tracker.total_cost() > 0
    assert len(tracker.summary()["by_model"]) == 1


def test_empty_input() -> None:
    from fluxrag.embedding.openai import OpenAIEmbedder
    embedder = OpenAIEmbedder(model="text-embedding-3-small")
    vectors = embedder.embed([])
    assert vectors == []
