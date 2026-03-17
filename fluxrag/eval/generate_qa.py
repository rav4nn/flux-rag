"""Synthetic QA pair generation from corpus chunks."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from fluxrag.core.schema import Chunk, QAPair
from fluxrag.llm.base import AbstractLLM

logger = logging.getLogger(__name__)

GENERATION_PROMPT = """You are generating question-answer pairs for evaluating a RAG system.

Given the following text chunk, generate exactly ONE question-answer pair.

Requirements:
- The question should be answerable ONLY from the given text
- The answer should be factual and directly supported by the text
- Vary difficulty: some easy (surface-level), some hard (requires reasoning)
- Do not ask "according to the text" or reference the text directly

Text chunk:
---
{chunk_text}
---

Respond in this exact JSON format (no other text):
{{"question": "...", "ground_truth": "...", "difficulty": "easy|medium|hard"}}"""


def generate_qa_pairs(
    chunks: list[Chunk],
    llm: AbstractLLM,
    target_count: int = 100,
    sample_size: int = 150,
    output_path: str | None = None,
) -> list[QAPair]:
    """Generate synthetic QA pairs from a sample of corpus chunks.

    Args:
        chunks: All chunks in the corpus.
        llm: LLM to use for generation.
        target_count: Minimum number of pairs to generate.
        sample_size: Number of chunks to sample (generates 1 pair per chunk).
        output_path: Optional path to write qa_pairs.jsonl.

    Returns:
        List of generated QAPair objects.
    """
    if len(chunks) < sample_size:
        sample = chunks
    else:
        # Stratified sampling by source_type if available
        by_source: dict[str, list[Chunk]] = {}
        for c in chunks:
            key = c.metadata.get("source_type", "unknown")
            by_source.setdefault(key, []).append(c)

        sample: list[Chunk] = []
        per_source = max(1, sample_size // len(by_source))
        for source_chunks in by_source.values():
            sample.extend(random.sample(source_chunks, min(per_source, len(source_chunks))))

        # Fill remaining slots randomly
        remaining = [c for c in chunks if c not in sample]
        if len(sample) < sample_size and remaining:
            sample.extend(random.sample(remaining, min(sample_size - len(sample), len(remaining))))

    pairs: list[QAPair] = []
    for chunk in sample:
        try:
            prompt = GENERATION_PROMPT.format(chunk_text=chunk.text[:2000])
            response = llm.generate(prompt, temperature=0.3)
            data = json.loads(response.strip())

            pair = QAPair(
                question=data["question"],
                ground_truth=data["ground_truth"],
                source_chunk_id=chunk.id,
                difficulty=data.get("difficulty", "medium"),
            )
            pairs.append(pair)
            logger.info("Generated QA pair %d/%d", len(pairs), target_count)

        except (json.JSONDecodeError, KeyError, Exception) as e:
            logger.warning("Failed to generate QA from chunk %s: %s", chunk.id, e)
            continue

        if len(pairs) >= target_count:
            break

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(pair.model_dump_json() + "\n")
        logger.info("Wrote %d QA pairs to %s", len(pairs), output_path)

    return pairs


def load_qa_pairs(path: str) -> list[QAPair]:
    """Load QA pairs from a JSONL file."""
    pairs: list[QAPair] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(QAPair.model_validate_json(line))
    return pairs
