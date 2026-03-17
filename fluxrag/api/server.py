"""FastAPI application factory and server runner."""

from __future__ import annotations

from fastapi import FastAPI

from fluxrag.api.routes import router


def create_app(config_path: str | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(title="FluxRAG", version="0.1.0")
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
