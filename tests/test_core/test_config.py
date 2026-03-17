"""Tests for FluxRAG config loading and validation."""

from pathlib import Path

from fluxrag.core.config import FluxRAGConfig


EXAMPLE_CONFIG = Path(__file__).parent.parent.parent / "examples" / "coffee" / "domain.yaml"


def test_config_loads_example_yaml() -> None:
    """The example coffee domain config should load and validate."""
    config = FluxRAGConfig.from_yaml(EXAMPLE_CONFIG)
    assert config.domain.name == "specialty_coffee"
    assert config.chunking.strategy == "semantic"
    assert config.retrieval.strategy == "hybrid_rerank"
    assert len(config.benchmark.embedding_models) == 8
    assert len(config.benchmark.llms) == 8
    assert len(config.benchmark.rerankers) == 4


def test_config_validates_chunking_strategy() -> None:
    """Invalid chunking strategy should raise a validation error."""
    import pytest

    with pytest.raises(Exception):
        FluxRAGConfig.model_validate({
            "domain": {"name": "test", "description": "test"},
            "corpus": {"sources": [{"path": "./data"}]},
            "chunking": {"strategy": "invalid"},
        })
