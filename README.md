# FluxRAG

A universal ingestion-to-evaluation RAG pipeline. Drop in files of any type — PDFs, YouTube transcripts, audio, images, HTML — and get an evaluated, benchmarked knowledge base with documented quality scores.

Built as the RAG backend for [coffeecoach.app](https://coffeecoach.app), with a specialty coffee corpus of 785+ YouTube transcripts and espresso technique guides.

## Why This Project Exists

Most RAG systems are built without evaluation. Teams pick an embedding model, a chunking strategy, and an LLM — then ship it without knowing if retrieval actually works. The result: hallucinations that go undetected, retrieval quality that degrades silently, and model choices based on blog posts instead of benchmarks.

FluxRAG takes the opposite approach: **build the evaluation harness first**, then optimise against it. Every chunking strategy, embedding model, retrieval method, and LLM is benchmarked on the same QA test set with documented quality/cost/latency tradeoffs.

## What Makes This Different

- **Evaluation-first** — LLM-judge harness with QA pairs runs before any optimisation. Every change is measured, not assumed.
- **Parameter sweep** — define a grid of chunking strategies, chunk sizes, and retrieval methods in `domain.yaml`. One command tests all combinations and writes a ranked comparison report.
- **Full benchmark matrices** — 4 sequential matrices (chunking x retrieval -> embeddings -> rerankers -> LLMs) produce a clear optimisation path without combinatorial explosion.
- **Universal ingestion** — 10 file types through a single parser router, plus native support for [youtube-rag-scraper](https://github.com/rav4nn/youtube-rag-scraper) JSON output.
- **Documented tradeoffs** — every design decision (ChromaDB vs Qdrant, dense vs hybrid, local vs API models) includes the reasoning and the cost.
- **Production-ready** — FastAPI server with health checks, Docker deployment, and cost tracking across all API calls.

## Architecture

```
[Any Source: PDFs, YouTube transcripts, audio, images, JSONL, HTML]
     |
Ingestion Layer (10 parsers + youtube-rag-scraper support)
     |
Chunking (fixed | sentence | semantic)
     |
Embedding (8 models: local sentence-transformers + OpenAI + Cohere + Voyage)
     |
Vector Store (ChromaDB) + BM25 Index
     |
Retrieval (dense | hybrid | hybrid + rerank)
     |
Reranking (5 models: cross-encoder + Cohere + Jina)
     |
LLM Generation (10 models: Claude + GPT + Gemini + Llama + Mistral + DeepSeek)
     |
Eval (LLM-judge: context recall, precision, faithfulness, answer relevancy)
     |
FastAPI Server (/query, /health, /eval, /benchmark)
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/rav4nn/flux-rag.git
cd flux-rag
python -m venv .venv
.venv/Scripts/Activate.ps1   # Windows
# source .venv/bin/activate  # Linux/Mac

# Install
pip install -e .

# Add API keys
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY (minimum required)

# Build the index (ingest + chunk + embed + store)
fluxrag build --config examples/coffee/domain.yaml

# Query it
fluxrag serve --config examples/coffee/domain.yaml
# POST http://localhost:8001/query {"question": "How do I dial in espresso?"}

# Evaluate against QA test set
fluxrag eval --config examples/coffee/domain.yaml --label "baseline"

# Run parameter sweep (tests all chunking/retrieval combos)
fluxrag sweep --config examples/coffee/domain.yaml --label "v1"
```

## CLI Commands

| Command | What it does |
|---|---|
| `fluxrag build --config domain.yaml` | Ingest all sources, chunk, embed, store to disk |
| `fluxrag serve --config domain.yaml` | Start FastAPI server on port 8001 |
| `fluxrag eval --config domain.yaml` | Score pipeline against QA pairs, save report |
| `fluxrag sweep --config domain.yaml` | Test all parameter combos from sweep config |
| `fluxrag benchmark --config domain.yaml` | Run full Matrix A benchmark |
| `fluxrag ingest --config domain.yaml` | Parse sources only (no embedding) |

## Configuration

Everything is configured in a single `domain.yaml` file:

```yaml
domain:
  name: "specialty_coffee"
  description: "Coffee brewing and espresso technique knowledge base"

corpus:
  sources:
    - path: "./data/legacy.jsonl"           # Pre-processed transcripts
      weight: 1.0
    - path: "./data/documents/"             # PDFs and text files
      types: ["pdf", "txt"]
      weight: 1.0
    - path: "./data/hoffmann.json"          # youtube-rag-scraper output
      type: "youtube_scraper"
      weight: 1.0

chunking:
  strategy: "semantic"        # fixed | sentence | semantic
  target_tokens: 400

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"

retrieval:
  strategy: "hybrid_rerank"   # dense | hybrid | hybrid_rerank
  top_k: 5
  rerank_candidates: 20
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  model: "deepseek/deepseek-chat"

eval:
  qa_pairs_path: "./eval/coffee_qa.jsonl"
  judge: "deepseek/deepseek-chat"

# Parameter sweep grid
sweep:
  chunking_strategies: ["fixed", "sentence", "semantic"]
  target_tokens: [200, 400, 600]
  retrieval_strategies: ["dense", "hybrid_rerank"]
  top_k: [5]
```

## Parameter Sweep

The sweep system tests every combination from your grid and writes a ranked comparison:

```bash
fluxrag sweep --config examples/coffee/domain.yaml --label "coffee-v1"
```

**Example output** (coffee corpus, 100 docs, 18 configs):

| # | Chunking | Tokens | Retrieval | Recall | Precision | Faithful | Relevancy | Avg |
|---|----------|--------|-----------|--------|-----------|----------|-----------|-----|
| 1 | fixed | 200 | dense | 0.187 | 0.293 | 0.953 | 0.500 | 0.483 |
| 2 | fixed | 400 | dense | 0.187 | 0.253 | 0.953 | 0.480 | 0.468 |
| 3 | fixed | 600 | dense | 0.187 | 0.267 | 0.920 | 0.480 | 0.463 |
| 4 | fixed | 600 | hybrid_rerank | 0.133 | 0.347 | 0.960 | 0.367 | 0.452 |

Winner: **fixed/200/dense/top_k=5** (avg 0.483). Faithfulness is high (0.95+) across all configs — the bottleneck is context recall.

Smart caching: chunks are computed once per (strategy, token_size) pair and reused across retrieval strategies. 18 configs = 9 chunk+embed passes, not 18.

## Supported Models

**Embeddings:** sentence-transformers (4 local), OpenAI (2), Cohere (1), Voyage (1)

**LLMs:** Claude Sonnet/Haiku, GPT-4o/mini, Gemini Flash/Pro, Llama 3.3 70B, Mistral Large, DeepSeek Chat/Reasoner

**Rerankers:** cross-encoder (3 local), Cohere (1), Jina (1)

## Ingestion Formats

| Format | Parser | Notes |
|---|---|---|
| PDF | pymupdf | Extracts text + metadata |
| DOCX | python-docx | Full document parsing |
| TXT/HTML | beautifulsoup4 | Plain text extraction |
| JSONL | Built-in | `{"id": "...", "text": "..."}` per line |
| CSV | pandas | Text column extraction |
| Images | pytesseract | OCR to text |
| Audio | faster-whisper | Speech-to-text transcription |
| YouTube URLs | youtube-transcript-api | Fetches transcripts from URL list |
| YouTube Scraper JSON | Built-in | Native [youtube-rag-scraper](https://github.com/rav4nn/youtube-rag-scraper) support |

## Project Structure

```
fluxrag/
  ingestion/     # 10 parsers + youtube-scraper support
  chunking/      # fixed, sentence, semantic strategies
  embedding/     # local + API embedders with factory
  retrieval/     # dense, hybrid, hybrid+rerank
  reranking/     # cross-encoder, Cohere, Jina
  llm/           # 10 LLM providers with factory
  eval/          # LLM-judge harness (4 RAGAS-style metrics)
  benchmark/     # Matrix runners + sweep system
  api/           # FastAPI server + routes
  core/          # Pipeline, config, schema
examples/
  coffee/        # Specialty coffee domain example
    domain.yaml  # Full configuration
    data/        # Corpus files (gitignored)
    eval/        # QA pairs + sweep reports
```

## Tech Stack

| Layer | Choice |
|---|---|
| Ingestion | pymupdf, python-docx, bs4, faster-whisper, yt-dlp, pytesseract |
| Chunking | spaCy, sentence-transformers (semantic similarity) |
| Embedding | sentence-transformers (local), OpenAI, Cohere, Voyage (API) |
| Vector Store | ChromaDB with persistent storage |
| Retrieval | rank_bm25 + dense search + cross-encoder reranking |
| LLM | Anthropic, OpenAI, Google, Groq, Mistral, DeepSeek SDKs |
| Evaluation | Custom LLM-judge harness (4 RAGAS-style metrics) |
| API | FastAPI + Uvicorn |
| Config | Pydantic + YAML |
| Deployment | Docker + docker-compose |

## License

MIT
