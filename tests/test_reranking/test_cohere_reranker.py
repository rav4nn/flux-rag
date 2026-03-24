"""Tests for Cohere reranker client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.benchmark.cost_tracker import CostTracker
from fluxrag.core.schema import Chunk, RetrievalResult


def _make_result(text: str, score: float) -> RetrievalResult:
    chunk = Chunk(text=text, document_id="doc1", index=0, metadata={})
    return RetrievalResult(chunk=chunk, score=score)


def _make_cohere_mock():
    """Create a mock cohere module with rerank support."""
    mock_module = MagicMock()

    def mock_rerank(query, documents, model, top_n):
        # Return results sorted by document length as a simple heuristic
        indexed = list(enumerate(documents))
        indexed.sort(key=lambda x: len(x[1]), reverse=True)
        items = []
        for rank, (idx, _doc) in enumerate(indexed[:top_n]):
            item = MagicMock()
            item.index = idx
            item.relevance_score = 1.0 - rank * 0.1
            items.append(item)
        resp = MagicMock()
        resp.results = items
        return resp

    mock_module.ClientV2.return_value.rerank.side_effect = mock_rerank
    return mock_module


@pytest.fixture(autouse=True)
def mock_cohere():
    mock_mod = _make_cohere_mock()
    with patch.dict(sys.modules, {"cohere": mock_mod}):
        if "fluxrag.reranking.cohere" in sys.modules:
            del sys.modules["fluxrag.reranking.cohere"]
        yield mock_mod


def test_rerank_returns_results() -> None:
    from fluxrag.reranking.cohere import CohereReranker
    reranker = CohereReranker(model="rerank-english-v3.0")
    results = [
        _make_result("short", 0.5),
        _make_result("a much longer document about coffee brewing", 0.3),
        _make_result("medium length text here", 0.4),
    ]
    reranked = reranker.rerank("coffee", results, top_k=2)
    assert len(reranked) == 2
    assert all(r.score > 0 for r in reranked)


def test_rerank_empty_results() -> None:
    from fluxrag.reranking.cohere import CohereReranker
    reranker = CohereReranker(model="rerank-english-v3.0")
    reranked = reranker.rerank("query", [], top_k=5)
    assert reranked == []


def test_original_rank_preserved() -> None:
    from fluxrag.reranking.cohere import CohereReranker
    reranker = CohereReranker(model="rerank-english-v3.0")
    results = [_make_result(f"Text number {i} with content", 0.5) for i in range(5)]
    reranked = reranker.rerank("query", results, top_k=3)
    assert all(r.original_rank >= 1 for r in reranked)


def test_model_name() -> None:
    from fluxrag.reranking.cohere import CohereReranker
    reranker = CohereReranker(model="rerank-english-v3.0")
    assert reranker.model_name == "cohere/rerank-english-v3.0"


def test_cost_tracking() -> None:
    from fluxrag.reranking.cohere import CohereReranker
    tracker = CostTracker()
    reranker = CohereReranker(model="rerank-english-v3.0", cost_tracker=tracker)
    results = [_make_result("text", 0.5)]
    reranker.rerank("query", results, top_k=1)
    assert tracker.total_cost() > 0
