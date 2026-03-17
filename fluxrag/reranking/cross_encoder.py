"""Local cross-encoder reranker using sentence-transformers."""

from __future__ import annotations

from fluxrag.core.schema import RerankedResult, RetrievalResult
from fluxrag.reranking.base import AbstractReranker


class CrossEncoderReranker(AbstractReranker):
    """Reranks retrieval results using a cross-encoder model.

    Scores each (query, candidate) pair independently.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        from sentence_transformers import CrossEncoder

        self._model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []

        # Build (query, candidate) pairs
        pairs = [(query, r.chunk.text) for r in results]
        scores = self._model.predict(pairs)

        # Combine with original rank info
        reranked: list[RerankedResult] = []
        for i, (result, score) in enumerate(zip(results, scores)):
            reranked.append(
                RerankedResult(
                    chunk=result.chunk,
                    score=float(score),
                    original_rank=i + 1,
                )
            )

        # Sort by reranker score descending, return top_k
        reranked.sort(key=lambda r: r.score, reverse=True)
        return reranked[:top_k]

    @property
    def model_name(self) -> str:
        return self._model_name
