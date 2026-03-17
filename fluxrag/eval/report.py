"""Evaluation report generator for benchmark matrices."""

from __future__ import annotations

from pathlib import Path

from fluxrag.core.schema import BenchmarkResult, EvalResult


def write_baseline_report(
    result: EvalResult,
    config_summary: dict[str, str],
    output_path: str,
) -> None:
    """Write the baseline evaluation report (Phase 2 deliverable)."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Baseline Evaluation Report",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for k, v in config_summary.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## RAGAS Scores",
        "",
        "| Metric | Score | Baseline Target | Status |",
        "|---|---|---|---|",
        f"| Context Recall | {result.context_recall:.3f} | > 0.70 | {'✅' if result.context_recall > 0.70 else '❌'} |",
        f"| Context Precision | {result.context_precision:.3f} | > 0.65 | {'✅' if result.context_precision > 0.65 else '❌'} |",
        f"| Faithfulness | {result.faithfulness:.3f} | > 0.80 | {'✅' if result.faithfulness > 0.80 else '❌'} |",
        f"| Answer Relevancy | {result.answer_relevancy:.3f} | > 0.75 | {'✅' if result.answer_relevancy > 0.75 else '❌'} |",
        "",
        "---",
        "",
        "*This baseline establishes the floor for all subsequent benchmark matrices.*",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_a_report(results: list[dict], output_path: str) -> None:
    """Write Matrix A: Chunking × Retrieval report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Matrix A: Chunking × Retrieval",
        "",
        "| Config | Chunking | Retrieval | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Latency (ms) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['config']} | {r['chunking']} | {r['retrieval']} | "
            f"{r['context_recall']:.3f} | {r['context_precision']:.3f} | "
            f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
            f"{r.get('latency_ms', '-')} |"
        )

    path.write_text("\n".join(lines), encoding="utf-8")
