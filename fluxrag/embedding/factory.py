"""Embedder factory — creates embedder instances from model name strings."""

from __future__ import annotations

from fluxrag.embedding.base import AbstractEmbedder


# All 8 models from PRD Matrix B
EMBEDDING_MODELS = {
    # Local models (sentence-transformers)
    "sentence-transformers/all-MiniLM-L6-v2": {"type": "local", "dimensions": 384},
    "BAAI/bge-large-en-v1.5": {"type": "local", "dimensions": 1024},
    "nomic-ai/nomic-embed-text-v1.5": {"type": "local", "dimensions": 768},
    "thenlper/gte-large": {"type": "local", "dimensions": 1024},
    # API models
    "openai/text-embedding-3-small": {"type": "openai", "dimensions": 1536},
    "openai/text-embedding-3-large": {"type": "openai", "dimensions": 3072},
    "cohere/embed-english-v3.0": {"type": "cohere", "dimensions": 1024},
    "voyage/voyage-3": {"type": "voyage", "dimensions": 1024},
}


def create_embedder(
    model_name: str,
    cost_tracker: object | None = None,
) -> AbstractEmbedder:
    """Create an embedder instance from a model name string.

    Args:
        model_name: Full model identifier (e.g. "openai/text-embedding-3-small").
        cost_tracker: Optional CostTracker for API cost tracking.

    Returns:
        An AbstractEmbedder instance.

    Raises:
        ValueError: If the model name is not recognized.
    """
    info = EMBEDDING_MODELS.get(model_name)
    if info is None:
        raise ValueError(
            f"Unknown embedding model: {model_name}. "
            f"Available: {list(EMBEDDING_MODELS.keys())}"
        )

    model_type = info["type"]

    if model_type == "local":
        from fluxrag.embedding.local import LocalEmbedder
        return LocalEmbedder(model_name)

    if model_type == "openai":
        from fluxrag.embedding.openai import OpenAIEmbedder
        # Strip provider prefix: "openai/text-embedding-3-small" -> "text-embedding-3-small"
        bare_model = model_name.split("/", 1)[1]
        return OpenAIEmbedder(model=bare_model, cost_tracker=cost_tracker)

    if model_type == "cohere":
        from fluxrag.embedding.cohere import CohereEmbedder
        bare_model = model_name.split("/", 1)[1]
        return CohereEmbedder(model=bare_model, cost_tracker=cost_tracker)

    if model_type == "voyage":
        from fluxrag.embedding.voyage import VoyageEmbedder
        bare_model = model_name.split("/", 1)[1]
        return VoyageEmbedder(model=bare_model, cost_tracker=cost_tracker)

    raise ValueError(f"Unknown embedder type: {model_type}")


def list_models() -> list[str]:
    """Return all available embedding model names."""
    return list(EMBEDDING_MODELS.keys())
