# FluxRAG

A universal ingestion-to-evaluation RAG pipeline. Drop in files of any type — PDFs, YouTube URLs, audio, images, HTML — and get an evaluated, benchmarked knowledge base with documented quality scores.

Built as the RAG backend for [coffeecoach.app](https://coffeecoach.app), replacing a hardcoded decision tree with retrieval-augmented generation.

## Why This Project Exists

Most RAG systems are built without evaluation. Teams pick an embedding model, a chunking strategy, and an LLM — then ship it without knowing if retrieval actually works. The result: hallucinations that go undetected, retrieval quality that degrades silently, and model choices based on blog posts instead of benchmarks.

FluxRAG takes the opposite approach: **build the evaluation harness first**, then optimise against it. Every chunking strategy, embedding model, retrieval method, and LLM is benchmarked on the same QA test set with documented quality/cost/latency tradeoffs.

## What Makes This Different

- **Evaluation-first** — RAGAS harness with 100+ QA pairs runs before any optimisation. Every change is measured, not assumed.
- **Full benchmark matrices** — 4 sequential matrices (chunking × retrieval → embeddings → rerankers → LLMs) produce a clear optimisation path without combinatorial explosion.
- **Universal ingestion** — 10 file types through a single parser router. Not locked to one data source or format.
- **Documented tradeoffs** — every design decision (ChromaDB vs Qdrant, dense vs hybrid, local vs API models) includes the reasoning and the cost.
- **Production deployment** — powers live features in coffeecoach.app, not just a notebook demo.

## Architecture

```
[Any Source: files, URLs, media]
     ↓
Ingestion Layer (10 parsers → unified Document schema)
     ↓
Chunking (fixed | sentence | semantic)
     ↓
Embedding (8 models benchmarked: local + OpenAI + Cohere + Voyage)
     ↓
Vector Store (ChromaDB dev / Qdrant prod) + BM25 Index
     ↓
Retrieval (dense | hybrid | hybrid + rerank)
     ↓
Reranking (4 models benchmarked: cross-encoder + Cohere + Jina)
     ↓
LLM Generation (8 models benchmarked: Claude + GPT + Gemini + Llama + Mistral)
     ↓
RAGAS Evaluation (context recall, precision, faithfulness, answer relevancy)
     ↓
FastAPI Endpoint → coffeecoach.app
```

The benchmark runs sequentially: **Matrix A** (chunking × retrieval) → **Matrix B** (embeddings) → **Matrix C** (rerankers) → **Matrix D** (LLMs). Each matrix fixes the winners from all previous matrices, keeping the total benchmark under a $50 budget while covering the full search space.

## Project Status

FluxRAG is in active development. See [PRD.md](PRD.md) for the full specification including:
- Detailed architecture and module map
- Benchmark matrix definitions
- Design tradeoffs and failure mode analysis
- Coffee domain case study
- Delivery phases

## Tech Stack

| Layer | Choice |
|---|---|
| Ingestion | pymupdf, python-docx, bs4, faster-whisper, yt-dlp, pytesseract |
| Chunking | spaCy, sentence-transformers (semantic similarity) |
| Embedding | sentence-transformers (local), OpenAI / Cohere / Voyage (API) |
| Vector Store | ChromaDB (dev), Qdrant (prod) |
| Retrieval | rank_bm25 + dense search + cross-encoder reranking |
| LLM | Anthropic, OpenAI, Google, Groq, Mistral SDKs |
| Evaluation | RAGAS |
| API | FastAPI + Uvicorn |
| Deployment | Docker + systemd + Nginx on Hetzner |

## License

MIT
