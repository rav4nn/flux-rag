"""Reranker factory — creates reranker instances from model name strings."""

from __future__ import annotations

from fluxrag.reranking.base import AbstractReranker


# All 4 reranker models from PRD Matrix C
RERANKER_MODELS = {
    # Local cross-encoders (sentence-transformers)
    "cross-encoder/ms-marco-MiniLM-L-6-v2": {"type": "local"},
    "BAAI/bge-reranker-v2-m3": {"type": "local"},
    "jinaai/jina-reranker-v2-base-multilingual": {"type": "local"},
    # API rerankers
    "cohere/rerank-english-v3.0": {"type": "cohere"},
    "jina/jina-reranker-v2-base-multilingual": {"type": "jina"},
}


def create_reranker(
    model_name: str,
    cost_tracker: object | None = None,
) -> AbstractReranker:
    """Create a reranker instance from a model name string.

    Args:
        model_name: Full model identifier (e.g. "cohere/rerank-english-v3.0").
        cost_tracker: Optional CostTracker for API cost tracking.

    Returns:
        An AbstractReranker instance.

    Raises:
        ValueError: If the model name is not recognized.
    """
    info = RERANKER_MODELS.get(model_name)
    if info is None:
        raise ValueError(
            f"Unknown reranker model: {model_name}. "
            f"Available: {list(RERANKER_MODELS.keys())}"
        )

    model_type = info["type"]

    if model_type == "local":
        from fluxrag.reranking.cross_encoder import CrossEncoderReranker
        return CrossEncoderReranker(model_name)

    if model_type == "cohere":
        from fluxrag.reranking.cohere import CohereReranker
        bare_model = model_name.split("/", 1)[1]
        return CohereReranker(model=bare_model, cost_tracker=cost_tracker)

    if model_type == "jina":
        from fluxrag.reranking.jina import JinaReranker
        bare_model = model_name.split("/", 1)[1]
        return JinaReranker(model=bare_model, cost_tracker=cost_tracker)

    raise ValueError(f"Unknown reranker type: {model_type}")


def list_rerankers() -> list[str]:
    """Return all available reranker model names."""
    return list(RERANKER_MODELS.keys())
