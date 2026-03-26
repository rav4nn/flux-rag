"""Sweep runner — evaluates a user-defined parameter grid and writes comparison reports."""

from __future__ import annotations

import csv
import datetime
import logging
import sys
import time
from pathlib import Path
from typing import Any

from fluxrag.chunking.base import AbstractChunker
from fluxrag.chunking.fixed import FixedChunker
from fluxrag.chunking.sentence import SentenceChunker
from fluxrag.chunking.semantic import SemanticChunker
from fluxrag.core.config import SweepConfig
from fluxrag.core.schema import Chunk, Document, EmbeddedChunk
from fluxrag.embedding.base import AbstractEmbedder
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.eval.generate_qa import load_qa_pairs
from fluxrag.eval.harness import EvalHarness
from fluxrag.llm.base import AbstractLLM
from fluxrag.reranking.cross_encoder import CrossEncoderReranker
from fluxrag.retrieval.dense import DenseRetriever
from fluxrag.retrieval.hybrid import HybridRetriever
from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever

logger = logging.getLogger(__name__)


class SweepRunner:
    """Iterates a parameter grid (chunking × tokens × retrieval × top_k) and scores each config."""

    def __init__(
        self,
        documents: list[Document],
        embedder: AbstractEmbedder,
        generator: AbstractLLM,
        judge: AbstractLLM,
        qa_pairs_path: str,
        sweep_config: SweepConfig,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        rerank_candidates: int = 20,
    ) -> None:
        self.documents = documents
        self.embedder = embedder
        self.generator = generator
        self.judge = judge
        self.qa_pairs = load_qa_pairs(qa_pairs_path)
        self.sweep = sweep_config
        self.reranker_model = reranker_model
        self.rerank_candidates = rerank_candidates
        self._reranker: CrossEncoderReranker | None = None

    def run(self) -> list[dict[str, Any]]:
        """Run every configuration in the grid. Returns list of result dicts."""
        configs = self._build_grid()
        total = len(configs)
        logger.info("[Sweep] %d configurations to evaluate", total)

        results: list[dict[str, Any]] = []

        # Cache: (chunking_strategy, target_tokens) → (chunks, store)
        chunk_cache: dict[tuple[str, int], tuple[list[Chunk], ChromaDBStore]] = {}

        for i, cfg in enumerate(configs, 1):
            cs = cfg["chunking"]
            tokens = cfg["target_tokens"]
            rs = cfg["retrieval"]
            tk = cfg["top_k"]

            self._print(f"\n{'='*60}")
            self._print(f"[{i}/{total}] {cs} / {tokens} tokens / {rs} / top_k={tk}")
            self._print(f"{'='*60}")

            cache_key = (cs, tokens)
            if cache_key not in chunk_cache:
                self._print(f"  Chunking {len(self.documents)} documents with '{cs}' strategy ({tokens} tokens)...")
                chunk_start = time.time()
                chunker = self._create_chunker(cs)
                chunks = self._chunk_all_with_progress(chunker, tokens)
                chunk_elapsed = time.time() - chunk_start
                self._print(f"  Chunked into {len(chunks)} chunks ({chunk_elapsed:.1f}s)")

                self._print(f"  Embedding {len(chunks)} chunks...")
                embed_start = time.time()
                store = self._build_store_with_progress(chunks, collection_name=f"sweep_{cs}_{tokens}")
                embed_elapsed = time.time() - embed_start
                self._print(f"  Embedded and stored ({embed_elapsed:.1f}s)")

                chunk_cache[cache_key] = (chunks, store)
            else:
                chunks, _ = chunk_cache[cache_key]
                self._print(f"  Reusing cached chunks for {cs}/{tokens} ({len(chunks)} chunks)")

            chunks, store = chunk_cache[cache_key]
            retriever = self._create_retriever(rs, chunks, store)

            self._print(f"  Evaluating {len(self.qa_pairs)} QA pairs...")
            start = time.time()
            harness = EvalHarness(
                retriever=retriever,
                generator=self.generator,
                judge=self.judge,
                top_k=tk,
            )
            eval_result = harness.evaluate(self.qa_pairs, progress_callback=self._eval_progress)
            elapsed = time.time() - start
            # Clear the progress line
            self._print("")

            avg = (
                eval_result.context_recall
                + eval_result.context_precision
                + eval_result.faithfulness
                + eval_result.answer_relevancy
            ) / 4

            results.append({
                "chunking": cs,
                "target_tokens": tokens,
                "retrieval": rs,
                "top_k": tk,
                "num_chunks": len(chunks),
                "context_recall": round(eval_result.context_recall, 3),
                "context_precision": round(eval_result.context_precision, 3),
                "faithfulness": round(eval_result.faithfulness, 3),
                "answer_relevancy": round(eval_result.answer_relevancy, 3),
                "avg_score": round(avg, 3),
                "latency_s": round(elapsed, 1),
            })

            self._print(
                f"  Results: Recall={eval_result.context_recall:.3f} "
                f"Prec={eval_result.context_precision:.3f} "
                f"Faith={eval_result.faithfulness:.3f} "
                f"Rel={eval_result.answer_relevancy:.3f} "
                f"Avg={avg:.3f} ({elapsed:.1f}s)"
            )

        # Clean up all cached stores
        for chunks, store in chunk_cache.values():
            store.reset()

        return results

    def _build_grid(self) -> list[dict[str, Any]]:
        """Generate all parameter combinations from SweepConfig."""
        grid: list[dict[str, Any]] = []
        for cs in self.sweep.chunking_strategies:
            for tokens in self.sweep.target_tokens:
                for rs in self.sweep.retrieval_strategies:
                    for tk in self.sweep.top_k:
                        grid.append({
                            "chunking": cs,
                            "target_tokens": tokens,
                            "retrieval": rs,
                            "top_k": tk,
                        })
        return grid

    @staticmethod
    def _print(msg: str) -> None:
        """Print progress directly to stderr so it always shows."""
        print(msg, file=sys.stderr, flush=True)

    @staticmethod
    def _eval_progress(current: int, total: int, question: str) -> None:
        """Callback for per-QA-pair eval progress."""
        truncated = question[:50] + "..." if len(question) > 50 else question
        print(f"\r  Eval [{current}/{total}] {truncated}            ", end="", file=sys.stderr, flush=True)

    def _chunk_all_with_progress(self, chunker: AbstractChunker, target_tokens: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        total = len(self.documents)
        for idx, doc in enumerate(self.documents, 1):
            if idx % 100 == 0 or idx == total:
                print(f"\r  Chunking [{idx}/{total}]", end="", file=sys.stderr, flush=True)
            chunks.extend(chunker.chunk(doc, target_tokens=target_tokens))
        print("", file=sys.stderr, flush=True)
        return chunks

    def _build_store_with_progress(self, chunks: list[Chunk], collection_name: str) -> ChromaDBStore:
        store = ChromaDBStore(collection_name=collection_name)
        store.reset()

        batch_size = 64
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        for batch_idx, i in enumerate(range(0, len(chunks), batch_size), 1):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(batch, embeddings)
            ]
            store.add(embedded)
            print(f"\r  Embedding [{batch_idx}/{total_batches}] ({i + len(batch)}/{len(chunks)} chunks)", end="", file=sys.stderr, flush=True)
        print("", file=sys.stderr, flush=True)
        return store

    def _create_chunker(self, strategy: str) -> AbstractChunker:
        if strategy == "fixed":
            return FixedChunker()
        elif strategy == "sentence":
            return SentenceChunker()
        elif strategy == "semantic":
            return SemanticChunker(self.embedder)
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _chunk_all(self, chunker: AbstractChunker, target_tokens: int) -> list[Chunk]:
        chunks: list[Chunk] = []
        for doc in self.documents:
            chunks.extend(chunker.chunk(doc, target_tokens=target_tokens))
        return chunks

    def _build_store(self, chunks: list[Chunk], collection_name: str) -> ChromaDBStore:
        store = ChromaDBStore(collection_name=collection_name)
        store.reset()

        batch_size = 64
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            embeddings = self.embedder.embed([c.text for c in batch])
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(batch, embeddings)
            ]
            store.add(embedded)
        return store

    def _create_retriever(self, strategy: str, chunks: list[Chunk], store: ChromaDBStore):
        if strategy == "dense":
            return DenseRetriever(self.embedder, store)
        elif strategy == "hybrid":
            return HybridRetriever(self.embedder, store, chunks)
        elif strategy == "hybrid_rerank":
            hybrid = HybridRetriever(self.embedder, store, chunks)
            if self._reranker is None:
                self._reranker = CrossEncoderReranker(self.reranker_model)
            return HybridRerankRetriever(hybrid, self._reranker, self.rerank_candidates)
        raise ValueError(f"Unknown retrieval strategy: {strategy}")


def write_sweep_report(
    results: list[dict[str, Any]],
    output_dir: Path,
    label: str = "",
) -> tuple[Path, Path]:
    """Write sweep results as markdown + CSV. Returns (md_path, csv_path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = f"_{label}" if label else ""

    # Sort by avg_score descending
    ranked = sorted(results, key=lambda r: r["avg_score"], reverse=True)
    winner = ranked[0] if ranked else None

    # Markdown report
    md_path = output_dir / f"sweep_{timestamp}{slug}.md"
    lines = [
        f"# FluxRAG Sweep Report{f' - {label}' if label else ''}",
        "",
        f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Configurations tested:** {len(results)}",
        "",
    ]

    if winner:
        lines += [
            "## Winner",
            "",
            f"**{winner['chunking']}** chunking / **{winner['target_tokens']}** tokens "
            f"/ **{winner['retrieval']}** retrieval / top_k={winner['top_k']}",
            f"Avg score: **{winner['avg_score']:.3f}**",
            "",
        ]

    lines += [
        "## Results",
        "",
        "| # | Chunking | Tokens | Retrieval | top_k | Chunks | Recall | Precision | Faithful | Relevancy | Avg | Time |",
        "|---|----------|--------|-----------|-------|--------|--------|-----------|----------|-----------|-----|------|",
    ]

    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['chunking']} | {r['target_tokens']} | {r['retrieval']} "
            f"| {r['top_k']} | {r['num_chunks']} "
            f"| {r['context_recall']:.3f} | {r['context_precision']:.3f} "
            f"| {r['faithfulness']:.3f} | {r['answer_relevancy']:.3f} "
            f"| {r['avg_score']:.3f} | {r['latency_s']}s |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # CSV report
    csv_path = output_dir / f"sweep_{timestamp}{slug}.csv"
    fieldnames = [
        "rank", "chunking", "target_tokens", "retrieval", "top_k", "num_chunks",
        "context_recall", "context_precision", "faithfulness", "answer_relevancy",
        "avg_score", "latency_s",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, r in enumerate(ranked, 1):
            writer.writerow({"rank": i, **r})

    return md_path, csv_path
