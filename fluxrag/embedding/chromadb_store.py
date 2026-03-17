"""ChromaDB vector store implementation."""

from __future__ import annotations

from typing import Any

import chromadb

from fluxrag.core.schema import Chunk, EmbeddedChunk, RetrievalResult
from fluxrag.embedding.store import AbstractVectorStore


class ChromaDBStore(AbstractVectorStore):
    """Vector store backed by ChromaDB. Zero-config for local development."""

    def __init__(
        self,
        collection_name: str = "fluxrag",
        persist_directory: str | None = None,
    ) -> None:
        if persist_directory:
            self._client = chromadb.PersistentClient(path=persist_directory)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, chunks: list[EmbeddedChunk]) -> None:
        if not chunks:
            return

        self._collection.add(
            ids=[c.chunk.id for c in chunks],
            embeddings=[c.embedding for c in chunks],
            documents=[c.chunk.text for c in chunks],
            metadatas=[self._sanitize_metadata(c.chunk.metadata) for c in chunks],
        )

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalResult]:
        where = self._build_where(filters) if filters else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()) if self._collection.count() > 0 else top_k,
            where=where,
        )

        if not results["ids"] or not results["ids"][0]:
            return []

        retrieval_results: list[RetrievalResult] = []
        for i, chunk_id in enumerate(results["ids"][0]):
            chunk = Chunk(
                id=chunk_id,
                text=results["documents"][0][i] if results["documents"] else "",
                document_id=results["metadatas"][0][i].get("document_id", "") if results["metadatas"] else "",
                index=int(results["metadatas"][0][i].get("index", 0)) if results["metadatas"] else 0,
                metadata=results["metadatas"][0][i] if results["metadatas"] else {},
            )
            # ChromaDB returns distances; cosine distance = 1 - similarity
            distance = results["distances"][0][i] if results["distances"] else 0.0
            score = 1.0 - distance
            retrieval_results.append(RetrievalResult(chunk=chunk, score=score))

        return retrieval_results

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        self._client.delete_collection(self._collection.name)
        self._collection = self._client.get_or_create_collection(
            name=self._collection.name,
            metadata={"hnsw:space": "cosine"},
        )

    def _sanitize_metadata(self, metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
        """ChromaDB only accepts primitive metadata values."""
        sanitized: dict[str, str | int | float | bool] = {}
        for k, v in metadata.items():
            if isinstance(v, (str, int, float, bool)):
                sanitized[k] = v
            else:
                sanitized[k] = str(v)
        return sanitized

    def _build_where(self, filters: dict[str, Any]) -> dict[str, Any] | None:
        """Convert simple key=value filters to ChromaDB where clause."""
        if not filters:
            return None
        if len(filters) == 1:
            k, v = next(iter(filters.items()))
            return {k: {"$eq": v}}
        return {"$and": [{k: {"$eq": v}} for k, v in filters.items()]}
