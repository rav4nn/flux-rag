"""OpenAI embedding models via the OpenAI SDK."""

from __future__ import annotations

import os

from fluxrag.embedding.base import AbstractEmbedder


# Pricing per 1M tokens (as of 2025)
OPENAI_PRICING = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}

# Default dimensions per model
OPENAI_DIMENSIONS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


class OpenAIEmbedder(AbstractEmbedder):
    """OpenAI text-embedding-3 models."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cost_tracker: object | None = None,
    ) -> None:
        from openai import OpenAI

        self._model = model
        self._cost_tracker = cost_tracker
        self._dimensions_val = OPENAI_DIMENSIONS.get(model, 1536)
        self._client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(model=self._model, input=texts)
        self._track_cost(texts)
        return [item.embedding for item in response.data]

    def embed_query(self, query: str) -> list[float]:
        return self.embed([query])[0]

    @property
    def model_name(self) -> str:
        return f"openai/{self._model}"

    @property
    def dimensions(self) -> int:
        return self._dimensions_val

    def _track_cost(self, texts: list[str]) -> None:
        if self._cost_tracker is None:
            return
        # Rough token estimate: 1 token ≈ 4 chars
        total_chars = sum(len(t) for t in texts)
        est_tokens = total_chars // 4
        price_per_m = OPENAI_PRICING.get(self._model, 0.02)
        cost = est_tokens * price_per_m / 1_000_000
        self._cost_tracker.record(
            provider="openai",
            model=self._model,
            input_tokens=est_tokens,
            output_tokens=0,
            cost_usd=cost,
        )
