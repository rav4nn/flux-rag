"""Voyage AI embedding models via the Voyage SDK."""

from __future__ import annotations

import os

from fluxrag.embedding.base import AbstractEmbedder


# Pricing per 1M tokens (as of 2025)
VOYAGE_PRICING = {
    "voyage-3": 0.06,
}

VOYAGE_DIMENSIONS = {
    "voyage-3": 1024,
}


class VoyageEmbedder(AbstractEmbedder):
    """Voyage AI voyage-3 models."""

    def __init__(
        self,
        model: str = "voyage-3",
        cost_tracker: object | None = None,
    ) -> None:
        import voyageai

        self._model = model
        self._cost_tracker = cost_tracker
        self._dimensions_val = VOYAGE_DIMENSIONS.get(model, 1024)
        self._client = voyageai.Client(api_key=os.environ.get("VOYAGE_API_KEY"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, model=self._model, input_type="document")
        self._track_cost(texts)
        return result.embeddings

    def embed_query(self, query: str) -> list[float]:
        result = self._client.embed([query], model=self._model, input_type="query")
        self._track_cost([query])
        return result.embeddings[0]

    @property
    def model_name(self) -> str:
        return f"voyage/{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    def _track_cost(self, texts: list[str]) -> None:
        if self._cost_tracker is None:
            return
        total_chars = sum(len(t) for t in texts)
        est_tokens = total_chars // 4
        price_per_m = VOYAGE_PRICING.get(self._model, 0.06)
        cost = est_tokens * price_per_m / 1_000_000
        self._cost_tracker.record(
            provider="voyage",
            model=self._model,
            input_tokens=est_tokens,
            output_tokens=0,
            cost_usd=cost,
        )
