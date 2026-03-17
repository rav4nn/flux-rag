"""Token counting and cost enforcement for benchmark runs."""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when the benchmark budget limit is exceeded."""


class CostTracker:
    """Tracks API costs across all benchmark runs and enforces budget limits."""

    def __init__(self, budget_limit_usd: float = 50.0) -> None:
        self.budget_limit_usd = budget_limit_usd
        self._records: list[dict[str, Any]] = []
        self._total_cost: float = 0.0

    def record(
        self,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record a single API call's cost."""
        self._records.append({
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost_usd,
        })
        self._total_cost += cost_usd
        logger.debug("Cost: $%.4f (%s/%s) | Total: $%.4f", cost_usd, provider, model, self._total_cost)

    def total_cost(self) -> float:
        return self._total_cost

    def check_budget(self) -> None:
        """Raise BudgetExceededError if total cost exceeds budget_limit_usd."""
        if self._total_cost >= self.budget_limit_usd:
            raise BudgetExceededError(
                f"Budget exceeded: ${self._total_cost:.2f} >= ${self.budget_limit_usd:.2f}"
            )

    def summary(self) -> dict[str, Any]:
        """Return a summary of costs by provider and model."""
        by_model: dict[str, float] = {}
        for r in self._records:
            key = f"{r['provider']}/{r['model']}"
            by_model[key] = by_model.get(key, 0.0) + r["cost_usd"]
        return {
            "total_cost_usd": self._total_cost,
            "budget_limit_usd": self.budget_limit_usd,
            "budget_remaining_usd": max(0, self.budget_limit_usd - self._total_cost),
            "calls": len(self._records),
            "by_model": by_model,
        }
