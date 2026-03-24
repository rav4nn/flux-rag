"""Cohere reranker via the Cohere SDK."""

from __future__ import annotations

import os

from fluxrag.core.schema import RerankedResult, RetrievalResult
from fluxrag.reranking.base import AbstractReranker


# $1 per 1,000 searches
COHERE_RERANK_COST_PER_SEARCH = 0.001


class CohereReranker(AbstractReranker):
    """Reranks using Cohere rerank-v3 API."""

    def __init__(
        self,
        model: str = "rerank-english-v3.0",
        cost_tracker: object | None = None,
    ) -> None:
        import cohere

        self._model = model
        self._cost_tracker = cost_tracker
        self._client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY"))

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []

        documents = [r.chunk.text for r in results]

        response = self._client.rerank(
            query=query,
            documents=documents,
            model=self._model,
            top_n=top_k,
        )

        self._track_cost()

        reranked: list[RerankedResult] = []
        for item in response.results:
            original_idx = item.index
            reranked.append(
                RerankedResult(
                    chunk=results[original_idx].chunk,
                    score=item.relevance_score,
                    original_rank=original_idx + 1,
                )
            )

        return reranked

    @property
    def model_name(self) -> str:
        return f"cohere/{self._model}"

    def _track_cost(self) -> None:
        if self._cost_tracker is None:
            return
        self._cost_tracker.record(
            provider="cohere",
            model=self._model,
            input_tokens=0,
            output_tokens=0,
            cost_usd=COHERE_RERANK_COST_PER_SEARCH,
        )
