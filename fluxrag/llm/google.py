"""Google Gemini LLM client (Gemini 2.5 Flash, Gemini 2.5 Pro)."""

from __future__ import annotations

import os

from fluxrag.llm.base import AbstractLLM


class GoogleLLM(AbstractLLM):
    """Google Gemini models via the Google GenAI SDK."""

    def __init__(self, model: str = "gemini-2.5-flash", max_tokens: int = 1024) -> None:
        from google import genai

        self._model = model
        self._max_tokens = max_tokens
        self._client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def generate(
        self,
        prompt: str,
        context: list[str] | None = None,
        **kwargs: object,
    ) -> str:
        from google.genai import types

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

        response = self._client.models.generate_content(
            model=self._model,
            contents=full_prompt,
            config=types.GenerateContentConfig(
                max_output_tokens=max_tokens,
                temperature=temperature,
            ),
        )
        return response.text

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider(self) -> str:
        return "google"
