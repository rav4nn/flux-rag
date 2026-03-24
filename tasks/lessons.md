# FluxRAG — Lessons Learned

Record anything that went wrong, took longer than expected, or would be done differently next time.

## Python 3.14 Compatibility
- **RAGAS** won't install (scikit-network needs older numpy). Built a custom LLM-judge eval harness instead. Works well and avoids the heavy dependency.
- **spaCy** won't install (pydantic v1/v2 conflict). Used regex-based sentence splitting instead — simpler and fast enough for our use case.

## Testing API Clients Without SDKs
- API embedder/reranker/LLM clients all use lazy imports (`from openai import OpenAI` inside `__init__`). This prevents import errors when SDKs aren't installed.
- Tests mock SDKs via `patch.dict(sys.modules, ...)` — the standard pattern for mocking lazy imports. Patching the attribute directly (`patch("module.Class")`) doesn't work because the name isn't in the module's namespace until the class is instantiated.

## Jina Reranker — No SDK Needed
- Used raw `urllib.request` for the Jina reranker API instead of adding a `jina` SDK dependency. The REST API is simple enough that a direct HTTP call is cleaner.

## Factory Pattern
- All providers (embedders, rerankers, LLMs) use a factory pattern with a central registry dict. This makes it trivial to add new models — just add an entry to the dict and implement the class.
- `create_embedder("openai/text-embedding-3-small")` is much cleaner than raw class instantiation scattered through the codebase.

## Sequential Benchmark Design
- Running all 4 matrices combinatorially would be 9 × 8 × 5 × 8 = 2,880 configs. The sequential design (A → B → C → D) reduces this to 9 + 8 + 5 + 8 = 30 configs — fits under $50 budget.

## FastAPI Lifecycle
- Used `lifespan` context manager (not deprecated `on_event`) for pipeline init on startup. The pipeline ingests + builds during startup, so the first request doesn't pay the cold-start cost.
- For tests, creating the app with `config_path=None` skips pipeline init — then mock `_pipeline` directly in the server module.
