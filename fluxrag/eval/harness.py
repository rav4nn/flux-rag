"""Evaluation harness — computes RAGAS-equivalent metrics using an LLM judge."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from fluxrag.core.schema import EvalResult, QAPair
from fluxrag.llm.base import AbstractLLM
from fluxrag.retrieval.base import AbstractRetriever

logger = logging.getLogger(__name__)


CONTEXT_RECALL_PROMPT = """You are evaluating a RAG system's retrieval quality.

Question: {question}
Ground Truth Answer: {ground_truth}

Retrieved Context:
{context}

Does the retrieved context contain the information needed to answer the question with the ground truth?
Score from 0.0 to 1.0 where:
- 1.0 = all information in the ground truth is present in the context
- 0.0 = none of the ground truth information is in the context

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


CONTEXT_PRECISION_PROMPT = """You are evaluating retrieval precision.

Question: {question}

Retrieved chunks (numbered):
{numbered_chunks}

For each chunk, is it relevant to answering the question?
Respond with ONLY a JSON object: {{"relevant_chunks": [<list of 1-based chunk numbers that are relevant>], "total": <total chunks>}}"""


FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to the provided context.

Question: {question}
Context: {context}
Answer: {answer}

Is the answer supported ONLY by the provided context? Does it contain any hallucinated information?
Score from 0.0 to 1.0 where:
- 1.0 = answer is completely supported by context, no hallucination
- 0.0 = answer contains significant hallucinated content

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an answer addresses the question.

Question: {question}
Answer: {answer}

Does the answer directly address the question asked?
Score from 0.0 to 1.0 where:
- 1.0 = answer directly and completely addresses the question
- 0.0 = answer is completely irrelevant to the question

Respond with ONLY a JSON object: {{"score": <float>, "reason": "<brief explanation>"}}"""


class EvalHarness:
    """Evaluates a retrieval + generation pipeline against a QA test set.

    Computes four RAGAS-equivalent metrics using an LLM judge:
    - Context Recall
    - Context Precision
    - Faithfulness
    - Answer Relevancy
    """

    def __init__(
        self,
        retriever: AbstractRetriever,
        generator: AbstractLLM,
        judge: AbstractLLM,
        top_k: int = 5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.judge = judge
        self.top_k = top_k

    def evaluate(
        self,
        qa_pairs: list[QAPair],
        output_path: str | None = None,
    ) -> EvalResult:
        """Run evaluation across all QA pairs and return aggregate scores."""
        recall_scores: list[float] = []
        precision_scores: list[float] = []
        faithfulness_scores: list[float] = []
        relevancy_scores: list[float] = []
        details: list[dict] = []

        for i, qa in enumerate(qa_pairs):
            logger.info("Evaluating QA pair %d/%d: %s", i + 1, len(qa_pairs), qa.question[:60])
            start = time.time()

            # Retrieve
            results = self.retriever.retrieve(qa.question, top_k=self.top_k)
            contexts = [r.chunk.text for r in results]

            # Generate
            answer = self.generator.generate(qa.question, context=contexts)

            # Score
            recall = self._score_context_recall(qa.question, qa.ground_truth, contexts)
            precision = self._score_context_precision(qa.question, contexts)
            faithfulness = self._score_faithfulness(qa.question, contexts, answer)
            relevancy = self._score_answer_relevancy(qa.question, answer)

            recall_scores.append(recall)
            precision_scores.append(precision)
            faithfulness_scores.append(faithfulness)
            relevancy_scores.append(relevancy)

            elapsed = time.time() - start
            details.append({
                "question": qa.question,
                "ground_truth": qa.ground_truth,
                "answer": answer,
                "context_recall": recall,
                "context_precision": precision,
                "faithfulness": faithfulness,
                "answer_relevancy": relevancy,
                "latency_s": round(elapsed, 2),
            })

        result = EvalResult(
            context_recall=self._mean(recall_scores),
            context_precision=self._mean(precision_scores),
            faithfulness=self._mean(faithfulness_scores),
            answer_relevancy=self._mean(relevancy_scores),
        )

        if output_path:
            self._write_report(result, details, output_path)

        return result

    def _score_context_recall(self, question: str, ground_truth: str, contexts: list[str]) -> float:
        prompt = CONTEXT_RECALL_PROMPT.format(
            question=question,
            ground_truth=ground_truth,
            context="\n\n---\n\n".join(contexts) if contexts else "(no context retrieved)",
        )
        return self._extract_score(prompt)

    def _score_context_precision(self, question: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        numbered = "\n".join(f"[{i+1}] {c[:500]}" for i, c in enumerate(contexts))
        prompt = CONTEXT_PRECISION_PROMPT.format(question=question, numbered_chunks=numbered)
        try:
            response = self.judge.generate(prompt)
            data = json.loads(response)
            relevant = len(data.get("relevant_chunks", []))
            total = data.get("total", len(contexts))
            return relevant / total if total > 0 else 0.0
        except Exception:
            return 0.5

    def _score_faithfulness(self, question: str, contexts: list[str], answer: str) -> float:
        prompt = FAITHFULNESS_PROMPT.format(
            question=question,
            context="\n\n---\n\n".join(contexts) if contexts else "(no context)",
            answer=answer,
        )
        return self._extract_score(prompt)

    def _score_answer_relevancy(self, question: str, answer: str) -> float:
        prompt = ANSWER_RELEVANCY_PROMPT.format(question=question, answer=answer)
        return self._extract_score(prompt)

    def _extract_score(self, prompt: str) -> float:
        """Send prompt to judge and extract the score field."""
        try:
            response = self.judge.generate(prompt)
            data = json.loads(response)
            score = float(data.get("score", 0.5))
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.5

    def _mean(self, scores: list[float]) -> float:
        return sum(scores) / len(scores) if scores else 0.0

    def _write_report(self, result: EvalResult, details: list[dict], output_path: str) -> None:
        """Write markdown evaluation report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        lines = [
            "# Evaluation Report",
            "",
            "## Aggregate Scores",
            "",
            "| Metric | Score |",
            "|---|---|",
            f"| Context Recall | {result.context_recall:.3f} |",
            f"| Context Precision | {result.context_precision:.3f} |",
            f"| Faithfulness | {result.faithfulness:.3f} |",
            f"| Answer Relevancy | {result.answer_relevancy:.3f} |",
            "",
            f"## Per-Question Details ({len(details)} questions)",
            "",
        ]

        for d in details:
            lines.extend([
                f"### Q: {d['question']}",
                f"- **Ground Truth:** {d['ground_truth'][:200]}",
                f"- **Answer:** {d['answer'][:200]}",
                f"- Recall: {d['context_recall']:.2f} | "
                f"Precision: {d['context_precision']:.2f} | "
                f"Faithfulness: {d['faithfulness']:.2f} | "
                f"Relevancy: {d['answer_relevancy']:.2f} | "
                f"Latency: {d['latency_s']}s",
                "",
            ])

        path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Wrote evaluation report to %s", output_path)
