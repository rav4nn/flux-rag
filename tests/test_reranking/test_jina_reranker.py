"""Tests for Jina reranker client."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.benchmark.cost_tracker import CostTracker
from fluxrag.core.schema import Chunk, RetrievalResult


def _make_result(text: str, score: float) -> RetrievalResult:
    chunk = Chunk(text=text, document_id="doc1", index=0, metadata={})
    return RetrievalResult(chunk=chunk, score=score)


def _mock_urlopen(req):
    """Mock urllib.request.urlopen to return fake rerank results."""
    body = json.loads(req.data.decode("utf-8"))
    docs = body["documents"]
    top_n = body.get("top_n", len(docs))

    # Return first top_n documents with decreasing scores
    results = []
    for i in range(min(top_n, len(docs))):
        results.append({"index": i, "relevance_score": 1.0 - i * 0.1})

    response_body = json.dumps({"results": results}).encode("utf-8")
    mock_resp = MagicMock()
    mock_resp.read.return_value = response_body
    mock_resp.__enter__ = lambda s: s
    mock_resp.__exit__ = MagicMock(return_value=False)
    return mock_resp


@pytest.fixture(autouse=True)
def mock_urlopen_fixture():
    with patch("fluxrag.reranking.jina.urllib.request.urlopen", side_effect=_mock_urlopen):
        yield


def test_rerank_returns_results() -> None:
    from fluxrag.reranking.jina import JinaReranker
    reranker = JinaReranker(model="jina-reranker-v2-base-multilingual")
    results = [
        _make_result("coffee brewing methods", 0.5),
        _make_result("espresso extraction", 0.3),
        _make_result("tea preparation", 0.4),
    ]
    reranked = reranker.rerank("coffee", results, top_k=2)
    assert len(reranked) == 2
    assert all(r.score > 0 for r in reranked)


def test_rerank_empty_results() -> None:
    from fluxrag.reranking.jina import JinaReranker
    reranker = JinaReranker(model="jina-reranker-v2-base-multilingual")
    reranked = reranker.rerank("query", [], top_k=5)
    assert reranked == []


def test_original_rank_preserved() -> None:
    from fluxrag.reranking.jina import JinaReranker
    reranker = JinaReranker(model="jina-reranker-v2-base-multilingual")
    results = [_make_result(f"Text {i}", 0.5) for i in range(5)]
    reranked = reranker.rerank("query", results, top_k=3)
    assert all(r.original_rank >= 1 for r in reranked)


def test_model_name() -> None:
    from fluxrag.reranking.jina import JinaReranker
    reranker = JinaReranker(model="jina-reranker-v2-base-multilingual")
    assert reranker.model_name == "jina/jina-reranker-v2-base-multilingual"


def test_cost_tracking() -> None:
    from fluxrag.reranking.jina import JinaReranker
    tracker = CostTracker()
    reranker = JinaReranker(model="jina-reranker-v2-base-multilingual", cost_tracker=tracker)
    results = [_make_result("text", 0.5)]
    reranker.rerank("query", results, top_k=1)
    assert tracker.total_cost() > 0
