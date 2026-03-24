"""Tests for OpenAI LLM client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_openai_mock():
    mock_module = MagicMock()

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Mocked response about coffee."
        return resp

    mock_module.OpenAI.return_value.chat.completions.create.side_effect = mock_create
    return mock_module


@pytest.fixture(autouse=True)
def mock_openai():
    mock_mod = _make_openai_mock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        if "fluxrag.llm.openai" in sys.modules:
            del sys.modules["fluxrag.llm.openai"]
        yield mock_mod


def test_generate_basic() -> None:
    from fluxrag.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o")
    result = llm.generate("What is espresso?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_context() -> None:
    from fluxrag.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o")
    result = llm.generate("What is espresso?", context=["Espresso is a concentrated coffee."])
    assert isinstance(result, str)


def test_model_name() -> None:
    from fluxrag.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o-mini")
    assert llm.model_name == "gpt-4o-mini"


def test_provider() -> None:
    from fluxrag.llm.openai import OpenAILLM
    llm = OpenAILLM(model="gpt-4o")
    assert llm.provider == "openai"
