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


def write_matrix_b_report(
    results: list[dict],
    matrix_a_winner: dict[str, str],
    cost_summary: dict | None,
    output_path: str,
) -> None:
    """Write Matrix B: Embedding Model benchmark report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Matrix B: Embedding Models",
        "",
        "## Fixed Configuration (Matrix A Winner)",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for k, v in matrix_a_winner.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Results",
        "",
        "| Model | Type | Dims | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Latency (ms) | Cost/1M tokens |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    best_score = -1.0
    best_model = ""

    for r in results:
        avg = (
            r["context_recall"] + r["context_precision"]
            + r["faithfulness"] + r["answer_relevancy"]
        ) / 4
        if avg > best_score and r.get("error") is None:
            best_score = avg
            best_model = r["model"]

        error_note = f" ⚠ {r['error']}" if r.get("error") else ""
        lines.append(
            f"| {r['model']} | {r['type']} | {r['dimensions']} | "
            f"{r['context_recall']:.3f} | {r['context_precision']:.3f} | "
            f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
            f"{r.get('latency_ms', '-')} | {r.get('cost_per_m_tokens', '-')} |{error_note}"
        )

    lines.extend([
        "",
        "## Winner",
        "",
        f"**{best_model}** (avg RAGAS: {best_score:.3f})",
        "",
    ])

    if cost_summary:
        lines.extend([
            "## Cost Summary",
            "",
            f"- Total spend: ${cost_summary.get('total_cost_usd', 0):.4f}",
            f"- Budget remaining: ${cost_summary.get('budget_remaining_usd', 0):.2f}",
            f"- API calls: {cost_summary.get('calls', 0)}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Matrix B winner feeds into Matrix C (reranker benchmark) as the fixed embedding model.*",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_c_report(
    results: list[dict],
    fixed_config: dict[str, str],
    cost_summary: dict | None,
    output_path: str,
) -> None:
    """Write Matrix C: Reranker benchmark report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Matrix C: Reranker Comparison",
        "",
        "## Fixed Configuration (Matrix A + B Winners)",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for k, v in fixed_config.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Results",
        "",
        "| Reranker | Type | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Latency (ms) | Cost/search |",
        "|---|---|---|---|---|---|---|---|",
    ])

    best_score = -1.0
    best_model = ""

    for r in results:
        avg = (
            r["context_recall"] + r["context_precision"]
            + r["faithfulness"] + r["answer_relevancy"]
        ) / 4
        if avg > best_score and r.get("error") is None:
            best_score = avg
            best_model = r["model"]

        error_note = f" ⚠ {r['error']}" if r.get("error") else ""
        lines.append(
            f"| {r['model']} | {r['type']} | "
            f"{r['context_recall']:.3f} | {r['context_precision']:.3f} | "
            f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
            f"{r.get('latency_ms', '-')} | {r.get('cost_per_search', '-')} |{error_note}"
        )

    lines.extend([
        "",
        "## Winner",
        "",
        f"**{best_model}** (avg RAGAS: {best_score:.3f})",
        "",
    ])

    if cost_summary:
        lines.extend([
            "## Cost Summary",
            "",
            f"- Total spend: ${cost_summary.get('total_cost_usd', 0):.4f}",
            f"- Budget remaining: ${cost_summary.get('budget_remaining_usd', 0):.2f}",
            f"- API calls: {cost_summary.get('calls', 0)}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Matrix C winner feeds into Matrix D (LLM benchmark) as the fixed reranker.*",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")


def write_matrix_d_report(
    results: list[dict],
    fixed_config: dict[str, str],
    cost_summary: dict | None,
    output_path: str,
) -> None:
    """Write Matrix D: LLM Generation benchmark report."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# Matrix D: LLM Generation Comparison",
        "",
        "## Fixed Configuration (Matrix A + B + C Winners)",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for k, v in fixed_config.items():
        lines.append(f"| {k} | {v} |")

    lines.extend([
        "",
        "## Results",
        "",
        "| LLM | Provider | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Latency (ms) | Input $/1M | Output $/1M |",
        "|---|---|---|---|---|---|---|---|---|",
    ])

    best_score = -1.0
    best_model = ""
    best_budget = ""
    best_budget_score = -1.0
    best_speed = ""
    best_latency = float("inf")

    for r in results:
        avg = (
            r["context_recall"] + r["context_precision"]
            + r["faithfulness"] + r["answer_relevancy"]
        ) / 4

        if avg > best_score and r.get("error") is None:
            best_score = avg
            best_model = r["model"]

        # Track best budget option (cheapest with decent quality)
        if r.get("error") is None and avg > 0.5:
            cost_str = r.get("input_cost_per_m", "$999")
            cost_val = float(cost_str.replace("$", ""))
            if cost_val < 1.0 and avg > best_budget_score:
                best_budget_score = avg
                best_budget = r["model"]

        # Track fastest
        latency = r.get("latency_ms", float("inf"))
        if r.get("error") is None and latency < best_latency:
            best_latency = latency
            best_speed = r["model"]

        error_note = f" ⚠ {r['error']}" if r.get("error") else ""
        lines.append(
            f"| {r['model']} | {r['provider']} | "
            f"{r['context_recall']:.3f} | {r['context_precision']:.3f} | "
            f"{r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} | "
            f"{r.get('latency_ms', '-')} | {r.get('input_cost_per_m', '-')} | "
            f"{r.get('output_cost_per_m', '-')} |{error_note}"
        )

    lines.extend([
        "",
        "## Winner",
        "",
        f"- **Best overall**: {best_model} (avg RAGAS: {best_score:.3f})",
    ])
    if best_budget:
        lines.append(f"- **Best budget**: {best_budget} (avg RAGAS: {best_budget_score:.3f})")
    if best_speed:
        lines.append(f"- **Fastest**: {best_speed} ({best_latency}ms/query)")

    lines.append("")

    if cost_summary:
        lines.extend([
            "## Cost Summary",
            "",
            f"- Total spend: ${cost_summary.get('total_cost_usd', 0):.4f}",
            f"- Budget remaining: ${cost_summary.get('budget_remaining_usd', 0):.2f}",
            f"- API calls: {cost_summary.get('calls', 0)}",
            "",
        ])

    lines.extend([
        "---",
        "",
        "*Matrix D completes the benchmark suite. Use the winning config for production deployment.*",
    ])

    path.write_text("\n".join(lines), encoding="utf-8")
