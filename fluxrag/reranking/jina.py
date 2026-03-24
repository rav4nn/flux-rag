"""Jina reranker via the Jina API."""

from __future__ import annotations

import json
import os
import urllib.request
import urllib.error

from fluxrag.core.schema import RerankedResult, RetrievalResult
from fluxrag.reranking.base import AbstractReranker


JINA_RERANK_URL = "https://api.jina.ai/v1/rerank"


class JinaReranker(AbstractReranker):
    """Reranks using Jina reranker-v2 API.

    Uses the Jina REST API directly (no SDK dependency).
    """

    def __init__(
        self,
        model: str = "jina-reranker-v2-base-multilingual",
        cost_tracker: object | None = None,
    ) -> None:
        self._model = model
        self._cost_tracker = cost_tracker
        self._api_key = os.environ.get("JINA_API_KEY", "")

    def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 5,
    ) -> list[RerankedResult]:
        if not results:
            return []

        documents = [{"text": r.chunk.text} for r in results]

        payload = json.dumps({
            "model": self._model,
            "query": query,
            "documents": documents,
            "top_n": top_k,
        }).encode("utf-8")

        req = urllib.request.Request(
            JINA_RERANK_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        self._track_cost()

        reranked: list[RerankedResult] = []
        for item in data["results"]:
            original_idx = item["index"]
            reranked.append(
                RerankedResult(
                    chunk=results[original_idx].chunk,
                    score=item["relevance_score"],
                    original_rank=original_idx + 1,
                )
            )

        return reranked

    @property
    def model_name(self) -> str:
        return f"jina/{self._model}"

    def _track_cost(self) -> None:
        if self._cost_tracker is None:
            return
        # Jina pricing: ~$0.02 per 1K tokens (rough estimate per search)
        self._cost_tracker.record(
            provider="jina",
            model=self._model,
            input_tokens=0,
            output_tokens=0,
            cost_usd=0.001,
        )
