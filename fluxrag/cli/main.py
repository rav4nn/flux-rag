"""FluxRAG CLI — ingest, build, eval, benchmark, serve."""

from __future__ import annotations

from pathlib import Path

import click
from dotenv import load_dotenv

from fluxrag.core.pipeline import Pipeline

# Load .env from project root (where the user runs commands from)
load_dotenv(Path.cwd() / ".env", override=False)


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
    """Ingest, chunk, embed, and store all documents."""
    pipeline = Pipeline.from_config(config)
    docs = pipeline.ingest()
    click.echo(f"Ingested {len(docs)} documents")
    pipeline.build()
    click.echo("Build complete")


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
@click.option("--label", default="", help="Label for this run, e.g. 'semantic-200' or 'fixed-400'")
def eval(config: str, label: str) -> None:
    """Run evaluation against the QA test set."""
    import datetime

    pipeline = Pipeline.from_config(config)
    pipeline.load()
    result = pipeline.evaluate()
    click.echo(result.summary())

    # Save report to eval/ folder next to config
    config_dir = Path(config).parent
    eval_dir = config_dir / "eval"
    eval_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{label}" if label else ""
    report_path = eval_dir / f"eval_{timestamp}{slug}.md"
    report_path.write_text(
        f"# Eval Report{f' — {label}' if label else ''}\n\n"
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        f"**Config:** {config}\n\n"
        f"## Scores\n\n"
        f"| Metric | Score |\n"
        f"|---|---|\n"
        f"| Context Recall | {result.context_recall:.3f} |\n"
        f"| Context Precision | {result.context_precision:.3f} |\n"
        f"| Faithfulness | {result.faithfulness:.3f} |\n"
        f"| Answer Relevancy | {result.answer_relevancy:.3f} |\n",
        encoding="utf-8",
    )
    click.echo(f"Report saved to {report_path}")


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def benchmark(config: str) -> None:
    """Run all configured benchmark matrices."""
    pipeline = Pipeline.from_config(config)
    result = pipeline.benchmark()
    click.echo(result.summary())


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
@click.option("--label", default="", help="Label for this sweep run")
def sweep(config: str, label: str) -> None:
    """Run parameter sweep — tests all chunking/retrieval combos from domain.yaml sweep section."""
    from fluxrag.benchmark.sweep import SweepRunner, write_sweep_report
    from fluxrag.embedding.factory import create_embedder
    from fluxrag.llm.factory import create_llm

    pipeline = Pipeline.from_config(config)
    docs = pipeline.ingest()
    click.echo(f"Ingested {len(docs)} documents")

    cfg = pipeline.config
    grid_size = (
        len(cfg.sweep.chunking_strategies)
        * len(cfg.sweep.target_tokens)
        * len(cfg.sweep.retrieval_strategies)
        * len(cfg.sweep.top_k)
    )
    click.echo(f"Sweep grid: {grid_size} configurations")
    click.echo(f"  Chunking: {cfg.sweep.chunking_strategies}")
    click.echo(f"  Tokens:   {cfg.sweep.target_tokens}")
    click.echo(f"  Retrieval:{cfg.sweep.retrieval_strategies}")
    click.echo(f"  top_k:    {cfg.sweep.top_k}")

    config_dir = Path(config).parent
    qa_path = str((config_dir / cfg.eval.qa_pairs_path).resolve())

    runner = SweepRunner(
        documents=docs,
        embedder=create_embedder(cfg.embedding.model),
        generator=create_llm(cfg.llm.model),
        judge=create_llm(cfg.eval.judge),
        qa_pairs_path=qa_path,
        sweep_config=cfg.sweep,
        reranker_model=cfg.retrieval.reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates=cfg.retrieval.rerank_candidates,
    )

    results = runner.run()

    eval_dir = config_dir / "eval"
    md_path, csv_path = write_sweep_report(results, eval_dir, label=label)

    click.echo("")
    click.echo(f"Sweep complete: {len(results)} configurations evaluated")
    if results:
        best = sorted(results, key=lambda r: r["avg_score"], reverse=True)[0]
        click.echo(
            f"Winner: {best['chunking']} / {best['target_tokens']} tokens "
            f"/ {best['retrieval']} / top_k={best['top_k']}  "
            f"(avg={best['avg_score']:.3f})"
        )
    click.echo(f"Report: {md_path}")
    click.echo(f"CSV:    {csv_path}")


@cli.command()
@click.option("--config", required=True, help="Path to domain.yaml config file")
def serve(config: str) -> None:
    """Start the FastAPI query server."""
    pipeline = Pipeline.from_config(config)
    pipeline.serve(config_path=config)


if __name__ == "__main__":
    cli()
