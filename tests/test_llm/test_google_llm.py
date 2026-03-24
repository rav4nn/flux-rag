"""Tests for Google Gemini LLM client."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest


def _make_google_mock():
    mock_genai = MagicMock()
    mock_types = MagicMock()

    def mock_generate(**kwargs):
        resp = MagicMock()
        resp.text = "Mocked Gemini response about coffee."
        return resp

    mock_genai.Client.return_value.models.generate_content.side_effect = mock_generate
    return mock_genai, mock_types


@pytest.fixture(autouse=True)
def mock_google():
    mock_genai, mock_types = _make_google_mock()
    mock_google_pkg = MagicMock()
    mock_google_pkg.genai = mock_genai
    with patch.dict(sys.modules, {
        "google": mock_google_pkg,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        if "fluxrag.llm.google" in sys.modules:
            del sys.modules["fluxrag.llm.google"]
        yield mock_genai


def test_generate_basic() -> None:
    from fluxrag.llm.google import GoogleLLM
    llm = GoogleLLM(model="gemini-2.5-flash")
    result = llm.generate("What is espresso?")
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_with_context() -> None:
    from fluxrag.llm.google import GoogleLLM
    llm = GoogleLLM(model="gemini-2.5-flash")
    result = llm.generate("What is espresso?", context=["Espresso is concentrated coffee."])
    assert isinstance(result, str)


def test_model_name() -> None:
    from fluxrag.llm.google import GoogleLLM
    llm = GoogleLLM(model="gemini-2.5-pro")
    assert llm.model_name == "gemini-2.5-pro"


def test_provider() -> None:
    from fluxrag.llm.google import GoogleLLM
    llm = GoogleLLM(model="gemini-2.5-flash")
    assert llm.provider == "google"
