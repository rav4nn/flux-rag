# FluxRAG — Active Tasks

## Current Phase: 2 — Baseline ✅ (pipeline built, awaiting coffee corpus)

Phase 0 complete. All ABCs defined, config models implemented, CLI/API stubs in place.
Phase 1 complete. All 9 parsers implemented + router + validator.
Phase 2 complete (code). Fixed chunking, local embedder, ChromaDB store, dense retrieval,
eval harness, QA generator, Anthropic LLM client, pipeline wiring. 61 tests passing.

To run baseline on coffee corpus:
1. Place coffee data in examples/coffee/data/
2. Set ANTHROPIC_API_KEY in .env
3. Run: fluxrag ingest --config examples/coffee/domain.yaml
4. Run: fluxrag build --config examples/coffee/domain.yaml
5. Generate QA: python -m fluxrag.eval.generate_qa (or manually create eval/coffee_qa.jsonl)
6. Run: fluxrag eval --config examples/coffee/domain.yaml

Phase 3 complete (code). Sentence chunking, semantic chunking, hybrid retrieval (BM25+dense+RRF),
cross-encoder reranking, hybrid+rerank retriever, Matrix A benchmark runner. 80 tests passing.

## Next: Phase 4 — Matrix B (Embedding Benchmark)

Implement embedding provider clients (OpenAI, Cohere, Voyage).
Run across all 8 models with Matrix A winner as fixed config.

Done when: `eval/matrix_b_embedding_report.md`.
