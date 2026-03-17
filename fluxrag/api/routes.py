"""FastAPI router with endpoint stubs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from fluxrag.api.models import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", corpus_size=0, index_ready=False)


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    raise HTTPException(status_code=501, detail="Query endpoint not yet implemented")


@router.get("/eval/latest")
async def eval_latest() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Eval endpoint not yet implemented")


@router.get("/benchmark/latest")
async def benchmark_latest() -> dict[str, str]:
    raise HTTPException(status_code=501, detail="Benchmark endpoint not yet implemented")
