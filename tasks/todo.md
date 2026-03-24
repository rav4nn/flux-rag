# FluxRAG — Active Tasks

## All Phases Complete (code) — awaiting benchmark runs + deployment

### Phase 0 ✅ — Scaffold
All ABCs defined, config models, CLI/API stubs.

### Phase 1 ✅ — Ingestion
9 parsers + router + validator.

### Phase 2 ✅ — Baseline Pipeline
Fixed chunking, local embedder, ChromaDB, dense retrieval, eval harness, Anthropic LLM client, pipeline wiring.

### Phase 3 ✅ — Matrix A (Chunking × Retrieval)
Sentence chunking, semantic chunking, hybrid retrieval (BM25+dense+RRF), cross-encoder reranking, hybrid+rerank, MatrixARunner.

### Phase 4 ✅ — Matrix B (Embedding Benchmark)
OpenAI/Cohere/Voyage embedder clients, embedder factory (8 models), MatrixBRunner.

### Phase 5 ✅ — Matrix C (Reranker Benchmark)
Cohere reranker, Jina reranker (REST API), reranker factory (5 models), MatrixCRunner.

### Phase 6 ✅ — Matrix D (LLM Benchmark)
OpenAI/Google/Groq/Mistral LLM clients, LLM factory (8 models, 5 providers), MatrixDRunner.

### Phase 7 ✅ — Production Deploy
FastAPI routes (query, health, eval, benchmark), Pipeline wired to factories, Dockerfile, docker-compose.yml.

93 tests passing.

## To Run Benchmarks

1. Place coffee data in `examples/coffee/data/`
2. Set API keys in `.env`:
   ```
   ANTHROPIC_API_KEY=...
   OPENAI_API_KEY=...
   CO_API_KEY=...
   VOYAGE_API_KEY=...
   GOOGLE_API_KEY=...
   GROQ_API_KEY=...
   MISTRAL_API_KEY=...
   JINA_API_KEY=...
   ```
3. Run sequentially: Matrix A → B → C → D (each winner feeds the next)
4. Reports generated in `eval/` directory

## To Deploy

```bash
docker build -t fluxrag:latest .
docker-compose up -d
# Or: fluxrag serve --config examples/coffee/domain.yaml
```
