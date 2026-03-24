"""FastAPI application factory and server runner."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from fluxrag.api.routes import router
from fluxrag.core.pipeline import Pipeline


# Module-level pipeline reference, set during app startup
_pipeline: Pipeline | None = None


def get_pipeline() -> Pipeline | None:
    """Return the current pipeline instance (used by routes)."""
    return _pipeline


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown lifecycle for the FastAPI app."""
    global _pipeline
    config_path = app.state.config_path
    if config_path:
        _pipeline = Pipeline.from_config(config_path)
        _pipeline.ingest()
        _pipeline.build()
    yield
    _pipeline = None


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="FluxRAG",
        version="0.1.0",
        description="Universal ingestion-to-evaluation RAG pipeline",
        lifespan=lifespan,
    )
    app.state.config_path = config_path
    app.include_router(router)
    return app


def run_server(
    config_path: str,
    host: str = "0.0.0.0",
    port: int = 8001,
) -> None:
    """Start the FluxRAG API server."""
    import uvicorn

    app = create_app(config_path)
    uvicorn.run(app, host=host, port=port)
