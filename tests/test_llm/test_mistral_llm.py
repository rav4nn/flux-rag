"""Tests for Mistral LLM client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_mistral_mock():
    mock_module = MagicMock()

    def mock_complete(**kwargs):
        resp = MagicMock()
        resp.choices = [MagicMock()]
        resp.choices[0].message.content = "Mocked Mistral response about coffee."
        return resp

    mock_module.Mistral.return_value.chat.complete.side_effect = mock_complete
    return mock_module


@pytest.fixture(autouse=True)
def mock_mistral():
    mock_mod = _make_mistral_mock()
    with patch.dict(sys.modules, {"mistralai": mock_mod}):
        if "fluxrag.llm.mistral" in sys.modules:
            del sys.modules["fluxrag.llm.mistral"]
        yield mock_mod


def test_generate_basic() -> None:
    from fluxrag.llm.mistral import MistralLLM
    llm = MistralLLM(model="mistral-large-latest")
    result = llm.generate("What is espresso?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_context() -> None:
    from fluxrag.llm.mistral import MistralLLM
    llm = MistralLLM(model="mistral-large-latest")
    result = llm.generate("What is espresso?", context=["Espresso is concentrated coffee."])
    assert isinstance(result, str)


def test_model_name() -> None:
    from fluxrag.llm.mistral import MistralLLM
    llm = MistralLLM(model="mistral-large-latest")
    assert llm.model_name == "mistral-large-latest"


def test_provider() -> None:
    from fluxrag.llm.mistral import MistralLLM
    llm = MistralLLM(model="mistral-large-latest")
    assert llm.provider == "mistral"
