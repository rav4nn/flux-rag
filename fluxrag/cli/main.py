"""FluxRAG CLI — ingest, build, eval, benchmark, serve."""

from __future__ import annotations

import click

from fluxrag.core.pipeline import Pipeline


@click.group()
def cli() -> None:
    """FluxRAG — Universal ingestion-to-evaluation RAG pipeline."""


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def ingest(config: str) -> None:
    """Parse all configured sources into unified documents."""
    pipeline = Pipeline.from_config(config)
    docs = pipeline.ingest()
    click.echo(f"Ingested {len(docs)} documents")


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def build(config: str) -> None:
    """Chunk, embed, and store all documents."""
    pipeline = Pipeline.from_config(config)
    pipeline.build()
    click.echo("Build complete")


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def eval(config: str) -> None:
    """Run RAGAS evaluation against the QA test set."""
    pipeline = Pipeline.from_config(config)
    result = pipeline.evaluate()
    click.echo(result.summary())


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def benchmark(config: str) -> None:
    """Run all configured benchmark matrices."""
    pipeline = Pipeline.from_config(config)
    result = pipeline.benchmark()
    click.echo(result.summary())


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def serve(config: str) -> None:
    """Start the FastAPI query server."""
    pipeline = Pipeline.from_config(config)
    pipeline.serve()


if __name__ == "__main__":
    cli()
