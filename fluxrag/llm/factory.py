"""LLM factory — creates LLM instances from model name strings."""

from __future__ import annotations

from fluxrag.llm.base import AbstractLLM


# All 8 LLM models from PRD Matrix D
LLM_MODELS = {
    # Anthropic
    "anthropic/claude-sonnet-4-20250514": {
        "type": "anthropic",
        "input_cost": 3.00,
        "output_cost": 15.00,
    },
    "anthropic/claude-haiku-4-20250414": {
        "type": "anthropic",
        "input_cost": 0.25,
        "output_cost": 1.25,
    },
    # OpenAI
    "openai/gpt-4o": {
        "type": "openai",
        "input_cost": 2.50,
        "output_cost": 10.00,
    },
    "openai/gpt-4o-mini": {
        "type": "openai",
        "input_cost": 0.15,
        "output_cost": 0.60,
    },
    # Google
    "google/gemini-2.5-flash": {
        "type": "google",
        "input_cost": 0.15,
        "output_cost": 0.60,
    },
    "google/gemini-2.5-pro": {
        "type": "google",
        "input_cost": 1.25,
        "output_cost": 10.00,
    },
    # Groq
    "groq/llama-3.3-70b-versatile": {
        "type": "groq",
        "input_cost": 0.59,
        "output_cost": 0.79,
    },
    # Mistral
    "mistral/mistral-large-latest": {
        "type": "mistral",
        "input_cost": 2.00,
        "output_cost": 6.00,
    },
    # DeepSeek
    "deepseek/deepseek-chat": {
        "type": "deepseek",
        "input_cost": 0.27,
        "output_cost": 1.10,
    },
    "deepseek/deepseek-reasoner": {
        "type": "deepseek",
        "input_cost": 0.55,
        "output_cost": 2.19,
    },
}


def create_llm(
    model_name: str,
    max_tokens: int = 1024,
) -> AbstractLLM:
    """Create an LLM instance from a model name string.

    Args:
        model_name: Full model identifier (e.g. "openai/gpt-4o").
        max_tokens: Maximum tokens for generation.

    Returns:
        An AbstractLLM instance.

    Raises:
        ValueError: If the model name is not recognized.
    """
    info = LLM_MODELS.get(model_name)
    if info is None:
        raise ValueError(
            f"Unknown LLM model: {model_name}. "
            f"Available: {list(LLM_MODELS.keys())}"
        )

    model_type = info["type"]
    bare_model = model_name.split("/", 1)[1]

    if model_type == "anthropic":
        from fluxrag.llm.anthropic import AnthropicLLM
        return AnthropicLLM(model=bare_model, max_tokens=max_tokens)

    if model_type == "openai":
        from fluxrag.llm.openai import OpenAILLM
        return OpenAILLM(model=bare_model, max_tokens=max_tokens)

    if model_type == "google":
        from fluxrag.llm.google import GoogleLLM
        return GoogleLLM(model=bare_model, max_tokens=max_tokens)

    if model_type == "groq":
        from fluxrag.llm.groq import GroqLLM
        return GroqLLM(model=bare_model, max_tokens=max_tokens)

    if model_type == "mistral":
        from fluxrag.llm.mistral import MistralLLM
        return MistralLLM(model=bare_model, max_tokens=max_tokens)

    if model_type == "deepseek":
        from fluxrag.llm.deepseek import DeepSeekLLM
        return DeepSeekLLM(model=bare_model, max_tokens=max_tokens)

    raise ValueError(f"Unknown LLM type: {model_type}")


def list_llms() -> list[str]:
    """Return all available LLM model names."""
    return list(LLM_MODELS.keys())


def get_cost_per_m(model_name: str) -> tuple[float, float]:
    """Return (input_cost, output_cost) per 1M tokens for the model."""
    info = LLM_MODELS.get(model_name, {})
    return (info.get("input_cost", 0.0), info.get("output_cost", 0.0))
