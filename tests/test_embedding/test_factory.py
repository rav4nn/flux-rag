"""Tests for embedder factory."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.embedding.factory import EMBEDDING_MODELS, create_embedder, list_models


def test_list_models_returns_all_eight() -> None:
    models = list_models()
    assert len(models) == 8
    assert "sentence-transformers/all-MiniLM-L6-v2" in models
    assert "openai/text-embedding-3-small" in models
    assert "cohere/embed-english-v3.0" in models
    assert "voyage/voyage-3" in models


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding model"):
        create_embedder("nonexistent/model")


def test_creates_local_embedder() -> None:
    mock_st_mod = MagicMock()
    mock_model = MagicMock()
    mock_model.get_sentence_embedding_dimension.return_value = 384
    mock_st_mod.SentenceTransformer.return_value = mock_model

    with patch.dict(sys.modules, {"sentence_transformers": mock_st_mod}):
        if "fluxrag.embedding.local" in sys.modules:
            del sys.modules["fluxrag.embedding.local"]
        embedder = create_embedder("sentence-transformers/all-MiniLM-L6-v2")
        assert "MiniLM" in embedder.model_name or "all-MiniLM" in embedder.model_name


def test_creates_openai_embedder() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        if "fluxrag.embedding.openai" in sys.modules:
            del sys.modules["fluxrag.embedding.openai"]
        embedder = create_embedder("openai/text-embedding-3-small")
        assert embedder.model_name == "openai/text-embedding-3-small"


def test_creates_cohere_embedder() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"cohere": mock_mod}):
        if "fluxrag.embedding.cohere" in sys.modules:
            del sys.modules["fluxrag.embedding.cohere"]
        embedder = create_embedder("cohere/embed-english-v3.0")
        assert embedder.model_name == "cohere/embed-english-v3.0"


def test_creates_voyage_embedder() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"voyageai": mock_mod}):
        if "fluxrag.embedding.voyage" in sys.modules:
            del sys.modules["fluxrag.embedding.voyage"]
        embedder = create_embedder("voyage/voyage-3")
        assert embedder.model_name == "voyage/voyage-3"


def test_embedding_models_registry_structure() -> None:
    for name, info in EMBEDDING_MODELS.items():
        assert "type" in info
        assert "dimensions" in info
        assert info["type"] in ("local", "openai", "cohere", "voyage")
        assert isinstance(info["dimensions"], int)
        assert info["dimensions"] > 0
