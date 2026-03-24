"""Tests for DeepSeek LLM client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_openai_mock():
    mock_module = MagicMock()

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Mocked DeepSeek response about coffee."
        return resp

    mock_module.OpenAI.return_value.chat.completions.create.side_effect = mock_create
    return mock_module


@pytest.fixture(autouse=True)
def mock_openai():
    mock_mod = _make_openai_mock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        if "fluxrag.llm.deepseek" in sys.modules:
            del sys.modules["fluxrag.llm.deepseek"]
        yield mock_mod


def test_generate_basic() -> None:
    from fluxrag.llm.deepseek import DeepSeekLLM
    llm = DeepSeekLLM(model="deepseek-chat")
    result = llm.generate("What is espresso?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_context() -> None:
    from fluxrag.llm.deepseek import DeepSeekLLM
    llm = DeepSeekLLM(model="deepseek-chat")
    result = llm.generate("What is espresso?", context=["Espresso is concentrated coffee."])
    assert isinstance(result, str)


def test_model_name() -> None:
    from fluxrag.llm.deepseek import DeepSeekLLM
    llm = DeepSeekLLM(model="deepseek-reasoner")
    assert llm.model_name == "deepseek-reasoner"


def test_provider() -> None:
    from fluxrag.llm.deepseek import DeepSeekLLM
    llm = DeepSeekLLM(model="deepseek-chat")
    assert llm.provider == "deepseek"


def test_uses_deepseek_base_url(mock_openai: MagicMock) -> None:
    from fluxrag.llm.deepseek import DeepSeekLLM, DEEPSEEK_BASE_URL
    llm = DeepSeekLLM(model="deepseek-chat")
    # Verify OpenAI client was created with DeepSeek base_url
    call_kwargs = mock_openai.OpenAI.call_args[1]
    assert call_kwargs["base_url"] == DEEPSEEK_BASE_URL
