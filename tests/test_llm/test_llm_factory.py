"""Tests for LLM factory."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from fluxrag.llm.factory import LLM_MODELS, create_llm, get_cost_per_m, list_llms


def test_list_llms_returns_all() -> None:
    models = list_llms()
    assert len(models) == 10
    assert "anthropic/claude-sonnet-4-20250514" in models
    assert "openai/gpt-4o" in models
    assert "google/gemini-2.5-flash" in models
    assert "groq/llama-3.3-70b-versatile" in models
    assert "mistral/mistral-large-latest" in models
    assert "deepseek/deepseek-chat" in models
    assert "deepseek/deepseek-reasoner" in models


def test_unknown_model_raises() -> None:
    with pytest.raises(ValueError, match="Unknown LLM model"):
        create_llm("nonexistent/model")


def test_creates_openai_llm() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        if "fluxrag.llm.openai" in sys.modules:
            del sys.modules["fluxrag.llm.openai"]
        llm = create_llm("openai/gpt-4o")
        assert llm.model_name == "gpt-4o"
        assert llm.provider == "openai"


def test_creates_google_llm() -> None:
    mock_genai = MagicMock()
    mock_types = MagicMock()
    mock_google = MagicMock()
    mock_google.genai = mock_genai
    with patch.dict(sys.modules, {
        "google": mock_google,
        "google.genai": mock_genai,
        "google.genai.types": mock_types,
    }):
        if "fluxrag.llm.google" in sys.modules:
            del sys.modules["fluxrag.llm.google"]
        llm = create_llm("google/gemini-2.5-flash")
        assert llm.model_name == "gemini-2.5-flash"
        assert llm.provider == "google"


def test_creates_groq_llm() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"groq": mock_mod}):
        if "fluxrag.llm.groq" in sys.modules:
            del sys.modules["fluxrag.llm.groq"]
        llm = create_llm("groq/llama-3.3-70b-versatile")
        assert llm.model_name == "llama-3.3-70b-versatile"
        assert llm.provider == "groq"


def test_creates_mistral_llm() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"mistralai": mock_mod}):
        if "fluxrag.llm.mistral" in sys.modules:
            del sys.modules["fluxrag.llm.mistral"]
        llm = create_llm("mistral/mistral-large-latest")
        assert llm.model_name == "mistral-large-latest"
        assert llm.provider == "mistral"


def test_creates_anthropic_llm() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"anthropic": mock_mod}):
        if "fluxrag.llm.anthropic" in sys.modules:
            del sys.modules["fluxrag.llm.anthropic"]
        llm = create_llm("anthropic/claude-sonnet-4-20250514")
        assert llm.model_name == "claude-sonnet-4-20250514"
        assert llm.provider == "anthropic"


def test_creates_deepseek_llm() -> None:
    mock_mod = MagicMock()
    with patch.dict(sys.modules, {"openai": mock_mod}):
        if "fluxrag.llm.deepseek" in sys.modules:
            del sys.modules["fluxrag.llm.deepseek"]
        llm = create_llm("deepseek/deepseek-chat")
        assert llm.model_name == "deepseek-chat"
        assert llm.provider == "deepseek"


def test_get_cost_per_m() -> None:
    input_cost, output_cost = get_cost_per_m("openai/gpt-4o")
    assert input_cost == 2.50
    assert output_cost == 10.00


def test_get_cost_per_m_unknown() -> None:
    input_cost, output_cost = get_cost_per_m("nonexistent/model")
    assert input_cost == 0.0
    assert output_cost == 0.0


def test_llm_models_registry_structure() -> None:
    for name, info in LLM_MODELS.items():
        assert "type" in info
        assert "input_cost" in info
        assert "output_cost" in info
        assert info["type"] in ("anthropic", "openai", "google", "groq", "mistral", "deepseek")
        assert isinstance(info["input_cost"], (int, float))
        assert isinstance(info["output_cost"], (int, float))
