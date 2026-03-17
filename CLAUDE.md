# FluxRAG — Claude Code Instructions

## Current Task
Check tasks/todo.md for the active task before doing anything else.

## Build Order
Always follow the phase order in PRD.md. Do not jump ahead.
Complete Phase 0 (scaffold) before implementing parsers.
Complete Phase 1 (ingestion) before running any evals.
Complete Matrix A before running embedding benchmarks.
Each matrix uses winners from the previous — do not skip.

## Key Constraints
- All parsers must implement the base.py AbstractParser interface
- All chunkers must implement the base.py AbstractChunker interface
- All embedders must implement the base.py AbstractEmbedder interface
- All rerankers must implement the base.py AbstractReranker interface
- All LLM clients must implement the base.py AbstractLLM interface
- All retrievers must implement a common AbstractRetriever interface
- Eval must run on the same 100-pair QA set across all benchmarks — never regenerate it mid-project
- Do not change qa_pairs.jsonl once Phase 2 baseline is locked in
- The cost_tracker.py must intercept every API call — never bypass it during benchmarks

## Testing
- Unit test every parser with at least 3 edge cases (empty file, minimal content, large file)
- Unit test every chunker with at least 3 edge cases (empty text, single sentence, very long text)
- Integration test the full pipeline end-to-end with a 10-chunk toy corpus before running on full corpus

## API Keys
- Store all API keys in .env (gitignored)
- Never hardcode keys in source files
- benchmark/cost_tracker.py enforces the budget_limit_usd from config

## Lessons
Check tasks/lessons.md before implementing anything you've tried before.
