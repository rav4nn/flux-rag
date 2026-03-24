"""DeepSeek LLM client (OpenAI-compatible API)."""

from __future__ import annotations

import os

from fluxrag.llm.base import AbstractLLM

DEEPSEEK_BASE_URL = "https://api.deepseek.com"


class DeepSeekLLM(AbstractLLM):
    """DeepSeek models via their OpenAI-compatible API."""

    def __init__(self, model: str = "deepseek-chat", max_tokens: int = 1024) -> None:
        from openai import OpenAI

        self._model = model
        self._max_tokens = max_tokens
        self._client = OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url=DEEPSEEK_BASE_URL,
        )

    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        max_tokens = int(kwargs.get("max_tokens", self._max_tokens))
        temperature = float(kwargs.get("temperature", 0.0))

        if context:
            context_block = "\n\n---\n\n".join(context)
            full_prompt = (
                f"Use the following context to answer the question. "
                f"Only use information from the context. If the context doesn't "
                f"contain the answer, say so.\n\n"
                f"Context:\n{context_block}\n\n"
                f"Question: {prompt}"
            )
        else:
            full_prompt = prompt

        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": full_prompt}],
        )
        return response.choices[0].message.content

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "deepseek"
