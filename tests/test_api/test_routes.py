"""Tests for FastAPI routes."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from fluxrag.api.server import create_app
from fluxrag.core.schema import Chunk, QueryResult, RetrievalResult


@pytest.fixture
def client():
    """Create a test client with no config (pipeline will be None)."""
    app = create_app(config_path=None)
    return TestClient(app)


def test_health_no_pipeline(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "not_initialized"
    assert data["index_ready"] is False


def test_query_no_pipeline(client: TestClient) -> None:
    resp = client.post("/query", json={"question": "What is espresso?"})
    assert resp.status_code == 503


def test_eval_latest_no_pipeline(client: TestClient) -> None:
    resp = client.get("/eval/latest")
    assert resp.status_code == 503


def test_benchmark_latest_no_pipeline(client: TestClient) -> None:
    resp = client.get("/benchmark/latest")
    assert resp.status_code == 503


def test_benchmark_run_returns_job_id(client: TestClient) -> None:
    # benchmark/run doesn't need pipeline (it queues a job)
    # But it does call _get_pipeline which will 503
    # Actually benchmark_run doesn't call _get_pipeline, so it should work
    resp = client.post("/benchmark/run", json={"matrices": ["embedding"], "budget_limit_usd": 10.0})
    assert resp.status_code == 200
    data = resp.json()
    assert "job_id" in data
    assert data["status"] == "queued"


def test_benchmark_status(client: TestClient) -> None:
    resp = client.get("/benchmark/status/fake-job-id")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_id"] == "fake-job-id"


def test_health_with_pipeline() -> None:
    """Test health check with a mocked pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_ready = True
    mock_pipeline.corpus_size = 42

    app = create_app(config_path=None)
    with patch("fluxrag.api.server._pipeline", mock_pipeline):
        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["corpus_size"] == 42
        assert data["index_ready"] is True


def test_query_with_pipeline() -> None:
    """Test query endpoint with a mocked pipeline."""
    mock_pipeline = MagicMock()
    mock_pipeline.is_ready = True
    chunk = Chunk(text="Espresso uses fine grounds.", document_id="doc1", index=0, metadata={})
    mock_pipeline.query.return_value = QueryResult(
        answer="Espresso is a concentrated coffee.",
        sources=[RetrievalResult(chunk=chunk, score=0.95)],
        latency_ms=150.0,
    )

    app = create_app(config_path=None)
    with patch("fluxrag.api.server._pipeline", mock_pipeline):
        client = TestClient(app)
        resp = client.post("/query", json={"question": "What is espresso?", "top_k": 3})
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Espresso is a concentrated coffee."
        assert len(data["sources"]) == 1
        assert data["sources"][0]["score"] == 0.95
        assert data["latency_ms"] == 150.0
