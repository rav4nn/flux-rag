"""Tests for Groq LLM client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_groq_mock():
    mock_module = MagicMock()

    def mock_create(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Mocked Groq response about coffee."
        return resp

    mock_module.Groq.return_value.chat.completions.create.side_effect = mock_create
    return mock_module


@pytest.fixture(autouse=True)
def mock_groq():
    mock_mod = _make_groq_mock()
    with patch.dict(sys.modules, {"groq": mock_mod}):
        if "fluxrag.llm.groq" in sys.modules:
            del sys.modules["fluxrag.llm.groq"]
        yield mock_mod


def test_generate_basic() -> None:
    from fluxrag.llm.groq import GroqLLM
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    result = llm.generate("What is espresso?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_context() -> None:
    from fluxrag.llm.groq import GroqLLM
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    result = llm.generate("What is espresso?", context=["Espresso is concentrated coffee."])
    assert isinstance(result, str)


def test_model_name() -> None:
    from fluxrag.llm.groq import GroqLLM
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    assert llm.model_name == "llama-3.3-70b-versatile"


def test_provider() -> None:
    from fluxrag.llm.groq import GroqLLM
    llm = GroqLLM(model="llama-3.3-70b-versatile")
    assert llm.provider == "groq"
