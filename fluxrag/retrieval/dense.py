"""Strategy 1: Dense vector search."""

from __future__ import annotations

from typing import Any

from fluxrag.core.schema import RetrievalResult
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.store import AbstractVectorStore
from fluxrag.retrieval.base import AbstractRetriever


class DenseRetriever(AbstractRetriever):
    """Pure dense vector search over the vector store.

    Embeds the query with the same model used for the corpus,
    then performs cosine similarity search.
    """

    def __init__(self, embedder: AbstractEmbedder, store: AbstractVectorStore) -> None:
        self._embedder = embedder
        self._store = store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        query_embedding = self._embedder.embed_query(query)
        return self._store.search(query_embedding, top_k=top_k, filters=filters)

    @property
    def strategy_name(self) -> str:
        return "dense"
