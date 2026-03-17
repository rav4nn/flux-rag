"""Local embedding models via sentence-transformers."""

from __future__ import annotations

from fluxrag.embedding.base import AbstractEmbedder


class LocalEmbedder(AbstractEmbedder):
    """Wraps sentence-transformers models for local embedding.

    Supports: all-MiniLM-L6-v2, BGE-large, nomic-embed-text, GTE-large.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
        from sentence_transformers import SentenceTransformer

        # Strip provider prefix if present
        self._model_name = model_name
        clean_name = model_name.split("/", 1)[-1] if "/" in model_name else model_name
        self._model = SentenceTransformer(clean_name)
        self._dimensions = self._model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        embeddings = self._model.encode(texts, show_progress_bar=False)
        return [e.tolist() for e in embeddings]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions
