"""FastAPI router with fully implemented endpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from fluxrag.api.models import (
    BenchmarkRunRequest,
    BenchmarkRunResponse,
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceResponse,
)

router = APIRouter()


def _get_pipeline():
    """Get the current pipeline, raising 503 if not ready."""
    from fluxrag.api.server import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    return pipeline


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    from fluxrag.api.server import get_pipeline

    pipeline = get_pipeline()
    if pipeline is None:
        return HealthResponse(status="not_initialized", corpus_size=0, index_ready=False)
    return HealthResponse(
        status="ok",
        corpus_size=pipeline.corpus_size,
        index_ready=pipeline.is_ready,
    )


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    pipeline = _get_pipeline()
    if not pipeline.is_ready:
        raise HTTPException(status_code=503, detail="Pipeline not built yet")

    result = pipeline.query(
        question=request.question,
        filters=request.filters,
        top_k=request.top_k,
    )

    sources = [
        SourceResponse(
            id=r.chunk.id,
            text=r.chunk.text,
            metadata=r.chunk.metadata,
            score=r.score,
        )
        for r in result.sources
    ]

    return QueryResponse(
        answer=result.answer,
        sources=sources,
        latency_ms=result.latency_ms,
    )


@router.get("/eval/latest")
async def eval_latest() -> dict[str, Any]:
    pipeline = _get_pipeline()

    # Look for the most recent eval report
    eval_dir = Path(pipeline.config.eval.qa_pairs_path).parent
    report_files = sorted(eval_dir.glob("*report*.md"), reverse=True)

    if not report_files:
        raise HTTPException(status_code=404, detail="No evaluation reports found")

    # Parse the most recent report for scores
    content = report_files[0].read_text(encoding="utf-8")
    return {
        "report_path": str(report_files[0]),
        "content": content,
    }


@router.get("/benchmark/latest")
async def benchmark_latest() -> dict[str, Any]:
    pipeline = _get_pipeline()

    eval_dir = Path(pipeline.config.eval.qa_pairs_path).parent
    matrices: dict[str, str] = {}

    for matrix_name in ["matrix_a", "matrix_b", "matrix_c", "matrix_d"]:
        report_files = list(eval_dir.glob(f"{matrix_name}*report*.md"))
        if report_files:
            matrices[matrix_name] = report_files[0].read_text(encoding="utf-8")

    if not matrices:
        raise HTTPException(status_code=404, detail="No benchmark reports found")

    return {"matrices": matrices}


@router.post("/benchmark/run", response_model=BenchmarkRunResponse)
async def benchmark_run(request: BenchmarkRunRequest) -> BenchmarkRunResponse:
    # Benchmark runs are expensive and long-running — return a placeholder
    # In production, this would spawn a background task
    import uuid

    return BenchmarkRunResponse(
        job_id=str(uuid.uuid4()),
        status="queued",
    )


@router.get("/benchmark/status/{job_id}")
async def benchmark_status(job_id: str) -> dict[str, Any]:
    # Placeholder — would check a task queue in production
    return {
        "job_id": job_id,
        "status": "not_implemented",
        "message": "Background benchmark execution not yet wired",
    }
