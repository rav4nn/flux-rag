"""Cohere embedding models via the Cohere SDK."""

from __future__ import annotations

import os

from fluxrag.embedding.base import AbstractEmbedder


# Pricing per 1M tokens (as of 2025)
COHERE_PRICING = {
    "embed-english-v3.0": 0.10,
}

COHERE_DIMENSIONS = {
    "embed-english-v3.0": 1024,
}


class CohereEmbedder(AbstractEmbedder):
    """Cohere embed-v3 models."""

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        cost_tracker: object | None = None,
    ) -> None:
        import cohere

        self._model = model
        self._cost_tracker = cost_tracker
        self._dimensions_val = COHERE_DIMENSIONS.get(model, 1024)
        self._client = cohere.ClientV2(api_key=os.environ.get("CO_API_KEY"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
            embedding_types=["float"],
        )
        self._track_cost(texts)
        return [list(e) for e in response.embeddings.float_]

    def embed_query(self, query: str) -> list[float]:
        response = self._client.embed(
            texts=[query],
            model=self._model,
            input_type="search_query",
            embedding_types=["float"],
        )
        self._track_cost([query])
        return list(response.embeddings.float_[0])

    @property
    def model_name(self) -> str:
        return f"cohere/{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    def _track_cost(self, texts: list[str]) -> None:
        if self._cost_tracker is None:
            return
        total_chars = sum(len(t) for t in texts)
        est_tokens = total_chars // 4
        price_per_m = COHERE_PRICING.get(self._model, 0.10)
        cost = est_tokens * price_per_m / 1_000_000
        self._cost_tracker.record(
            provider="cohere",
            model=self._model,
            input_tokens=est_tokens,
            output_tokens=0,
            cost_usd=cost,
        )
