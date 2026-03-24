"""Tests for Voyage AI embedding client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.benchmark.cost_tracker import CostTracker


def _make_voyage_mock():
    """Create a mock voyageai module."""
    mock_module = MagicMock()

    def make_response(texts, **kwargs):
        resp = MagicMock()
        resp.embeddings = [[0.1] * 1024 for _ in texts]
        return resp

    mock_module.Client.return_value.embed.side_effect = make_response
    return mock_module


@pytest.fixture(autouse=True)
def mock_voyage():
    """Inject mock voyageai module."""
    mock_mod = _make_voyage_mock()
    with patch.dict(sys.modules, {"voyageai": mock_mod}):
        if "fluxrag.embedding.voyage" in sys.modules:
            del sys.modules["fluxrag.embedding.voyage"]
        yield mock_mod


def test_embed_returns_correct_dimensions() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    embedder = VoyageEmbedder(model="voyage-3")
    vectors = embedder.embed(["test sentence"])
    assert len(vectors) == 1
    assert len(vectors[0]) == 1024


def test_embed_batch() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    embedder = VoyageEmbedder(model="voyage-3")
    texts = ["coffee is great", "espresso extraction", "grind size matters"]
    vectors = embedder.embed(texts)
    assert len(vectors) == 3


def test_embed_query_uses_query_type() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    embedder = VoyageEmbedder(model="voyage-3")
    vec = embedder.embed_query("what is espresso?")
    assert len(vec) == 1024
    client = embedder._client
    call_kwargs = client.embed.call_args_list[-1][1]
    assert call_kwargs["input_type"] == "query"


def test_model_name() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    embedder = VoyageEmbedder(model="voyage-3")
    assert embedder.model_name == "voyage/voyage-3"


def test_cost_tracking() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    tracker = CostTracker()
    embedder = VoyageEmbedder(model="voyage-3", cost_tracker=tracker)
    embedder.embed(["hello world"])
    assert tracker.total_cost() > 0


def test_empty_input() -> None:
    from fluxrag.embedding.voyage import VoyageEmbedder
    embedder = VoyageEmbedder(model="voyage-3")
    vectors = embedder.embed([])
    assert vectors == []
