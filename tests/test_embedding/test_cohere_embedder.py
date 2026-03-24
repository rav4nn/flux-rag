"""Tests for Cohere embedding client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.benchmark.cost_tracker import CostTracker


def _make_cohere_mock():
    """Create a mock cohere module."""
    mock_module = MagicMock()

    def make_response(texts, **kwargs):
        embeddings_obj = MagicMock()
        embeddings_obj.float_ = [[0.1] * 1024 for _ in texts]
        resp = MagicMock()
        resp.embeddings = embeddings_obj
        return resp

    mock_module.ClientV2.return_value.embed.side_effect = make_response
    return mock_module


@pytest.fixture(autouse=True)
def mock_cohere():
    """Inject mock cohere module."""
    mock_mod = _make_cohere_mock()
    with patch.dict(sys.modules, {"cohere": mock_mod}):
        if "fluxrag.embedding.cohere" in sys.modules:
            del sys.modules["fluxrag.embedding.cohere"]
        yield mock_mod


def test_embed_returns_correct_dimensions() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    embedder = CohereEmbedder(model="embed-english-v3.0")
    vectors = embedder.embed(["test sentence"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


def test_embed_batch() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    embedder = CohereEmbedder(model="embed-english-v3.0")
    texts = ["coffee is great", "espresso extraction", "grind size matters"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3


def test_embed_query_uses_search_query_type() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    embedder = CohereEmbedder(model="embed-english-v3.0")
    vec = embedder.embed_query("what is espresso?")
    assert len(vec) == 1024
    # Verify input_type="search_query" was used
    client = embedder._client
    call_kwargs = client.embed.call_args_list[-1][1]
    assert call_kwargs["input_type"] == "search_query"


def test_model_name() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    embedder = CohereEmbedder(model="embed-english-v3.0")
    assert embedder.model_name == "cohere/embed-english-v3.0"


def test_cost_tracking() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    tracker = CostTracker()
    embedder = CohereEmbedder(model="embed-english-v3.0", cost_tracker=tracker)
    embedder.embed(["hello world"])
    assert tracker.total_cost() > 0


def test_empty_input() -> None:
    from fluxrag.embedding.cohere import CohereEmbedder
    embedder = CohereEmbedder(model="embed-english-v3.0")
    vectors = embedder.embed([])
    assert vectors == []
