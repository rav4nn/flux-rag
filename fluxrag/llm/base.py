"""Abstract LLM interface. All LLM clients implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod


class AbstractLLM(ABC):
    """Base interface for all LLM clients.

    Covers Anthropic, OpenAI, Google, Groq, and Mistral providers.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        """Generate a response given a prompt and optional retrieved context.

        Args:
            prompt: The user's question or instruction.
            context: Optional list of retrieved text chunks to include as context.
            **kwargs: Model-specific options (e.g., temperature, max_tokens).

        Returns:
            Generated text response.
        """
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...

    @property
    @abstractmethod
    def provider(self) -> str:
        """Return the provider name (e.g., 'anthropic', 'openai')."""
        ...
