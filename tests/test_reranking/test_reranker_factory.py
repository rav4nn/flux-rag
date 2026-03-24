"""Tests for reranker factory."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.reranking.factory import RERANKER_MODELS, create_reranker, list_rerankers


def test_list_rerankers_returns_all() -> None:
    models = list_rerankers()
    assert len(models) == 5
    assert "cross-encoder/ms-marco-MiniLM-L-6-v2" in models
    assert "cohere/rerank-english-v3.0" in models
    assert "jina/jina-reranker-v2-base-multilingual" in models


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reranker model"):
        create_reranker("nonexistent/model")


def test_creates_cohere_reranker() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"cohere": mock_mod}):
        if "fluxrag.reranking.cohere" in sys.modules:
            del sys.modules["fluxrag.reranking.cohere"]
        reranker = create_reranker("cohere/rerank-english-v3.0")
        assert reranker.model_name == "cohere/rerank-english-v3.0"


def test_creates_jina_reranker() -> None:
    reranker = create_reranker("jina/jina-reranker-v2-base-multilingual")
    assert reranker.model_name == "jina/jina-reranker-v2-base-multilingual"


def test_creates_local_reranker() -> None:
    mock_st = MagicMock()
    with patch.dict(sys.modules, {"sentence_transformers": mock_st}):
        if "fluxrag.reranking.cross_encoder" in sys.modules:
            del sys.modules["fluxrag.reranking.cross_encoder"]
        reranker = create_reranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
        assert "ms-marco" in reranker.model_name


def test_reranker_models_registry_structure() -> None:
    for name, info in RERANKER_MODELS.items():
        assert "type" in info
        assert info["type"] in ("local", "cohere", "jina")
