"""Tests for cross-encoder reranker."""

import pytest

from fluxrag.core.schema import Chunk, RetrievalResult
from fluxrag.reranking.cross_encoder import CrossEncoderReranker


@pytest.fixture(scope="module")
def reranker() -> CrossEncoderReranker:
    return CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")


def _make_result(text: str, score: float) -> RetrievalResult:
    chunk = Chunk(text=text, document_id="doc1", index=0, metadata={})
    return RetrievalResult(chunk=chunk, score=score)


def test_reranks_by_relevance(reranker: CrossEncoderReranker) -> None:
    query = "What grind size for espresso?"
    results = [
        _make_result("Python is a programming language.", 0.9),  # High initial score, irrelevant
        _make_result("Espresso needs a fine grind like table salt.", 0.3),  # Low initial, relevant
        _make_result("Water temperature affects extraction.", 0.5),
    ]

    reranked = reranker.rerank(query, results, top_k=3)
    assert len(reranked) == 3
    # Espresso grind chunk should be ranked first after reranking
    assert "grind" in reranked[0].chunk.text.lower() or "espresso" in reranked[0].chunk.text.lower()


def test_top_k_limits_output(reranker: CrossEncoderReranker) -> None:
    results = [_make_result(f"Text {i}", 0.5) for i in range(10)]
    reranked = reranker.rerank("query", results, top_k=3)
    assert len(reranked) == 3


def test_empty_results(reranker: CrossEncoderReranker) -> None:
    reranked = reranker.rerank("query", [], top_k=5)
    assert reranked == []


def test_original_rank_preserved(reranker: CrossEncoderReranker) -> None:
    results = [_make_result(f"Text {i}", 0.5) for i in range(3)]
    reranked = reranker.rerank("query", results, top_k=3)
    assert all(r.original_rank >= 1 for r in reranked)


def test_model_name(reranker: CrossEncoderReranker) -> None:
    assert "ms-marco" in reranker.model_name
