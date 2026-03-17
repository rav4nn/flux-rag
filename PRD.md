# FluxRAG — Product Requirements Document

> **Version:** 2.0 | **Author:** Hardeep (Metasmic) | **Date:** March 2026
> **Status:** Active — use this file to drive Claude Code sessions
> **Legacy companion:** [youtube-rag-scraper](https://github.com/rav4nn/youtube-rag-scraper) (now optional — FluxRAG handles YouTube natively)

---

## 0. Problem Statement

Most RAG systems fail silently:

- **No evaluation** — cannot measure improvements
- **Hardcoded pipelines** — not reusable across domains
- **No benchmarking** — unclear model tradeoffs
- **Brittle data ingestion** — requires manual preprocessing

This leads to hallucinations in production, poor retrieval quality, wasted API cost, and non-reproducible systems.

FluxRAG solves this by introducing an evaluation-first, modular RAG pipeline that can ingest arbitrary data sources and produce measurable improvements.

---

## 1. What Is FluxRAG

FluxRAG is a universal ingestion-to-evaluation pipeline that accepts raw files of any type — PDFs, documents, text, HTML, images, audio, YouTube videos — and produces an evaluated, optimised, production-ready RAG knowledge base with documented quality scores and model benchmark results.

It does not care where the data came from or what format it's in. Drop in a folder of PDFs, a list of YouTube URLs, a directory of audio recordings, or a legacy JSONL dump — FluxRAG's ingestion layer parses everything into a unified document schema, then handles chunking, embedding, vector store population, hybrid retrieval, reranking, multi-model benchmarking, evaluation, and a FastAPI query endpoint.

```
[Any Source: files, URLs, media]
     ↓
FluxRAG Ingestion Layer (parsers + extractors)
     ↓
Unified Document Schema (internal)
     ↓
Chunking → Embedding → Vector Store
     ↓
Eval Harness (RAGAS) + Model Benchmarking
     ↓
Optimised FastAPI Endpoint
     ↓
Application (e.g. coffeecoach.app)
```

---

## 2. Goals

### 2.1 Technical Goals
- Accept raw files from any source — PDF, DOCX, TXT, Markdown, CSV, HTML, JSONL, images (OCR), audio (transcription), YouTube URLs (transcript extraction)
- Implement and benchmark 3 chunking strategies
- Implement and benchmark 3 retrieval strategies
- Benchmark 8 embedding models across open-source and commercial providers
- Benchmark 8 LLMs for generation and evaluation quality/cost/latency tradeoffs
- Benchmark 4 reranker models across local and API-based options
- Build a RAGAS evaluation harness with a synthetic QA test set (100+ pairs)
- Produce model selection matrices showing quality, cost, and latency tradeoffs
- Expose a FastAPI query endpoint deployable on any Linux server

### 2.2 Portfolio Goals
- Demonstrate AI engineering depth beyond "LLM wrapper" projects
- Show eval-first thinking — baseline before optimising, not after
- Demonstrate model evaluation rigour across embedding, generation, and reranking layers
- Connect to a live product (coffeecoach.app) as a real-world case study
- Show universal ingestion capability — not locked to a single data source

### 2.3 What This Is Not
- Not a chatbot UI (the consuming application handles that)
- Not a model training or fine-tuning pipeline
- Not a document management system — it processes files but does not manage versions
- Not a managed cloud service

---

## 3. Input Contract

### 3.1 Supported File Types

FluxRAG natively ingests the following formats. Each parser converts raw files into the unified Document schema.

| Format | Parser / Library | Notes |
|---|---|---|
| PDF | `pymupdf` (PyMuPDF) | Text extraction; falls back to OCR for scanned pages |
| TXT | built-in | Direct read, encoding detection via `chardet` |
| Markdown | built-in | Strip formatting or preserve structure |
| DOCX | `python-docx` | Paragraphs + tables extraction |
| CSV / TSV | `pandas` | Each row or N-row group becomes a document |
| HTML | `beautifulsoup4` + `trafilatura` | Main content extraction, boilerplate removal |
| JSONL | built-in | Legacy format, direct schema conformance |
| Images (PNG/JPG/TIFF) | `pytesseract` or Claude Vision API | OCR to text |
| Audio (MP3/WAV/M4A) | `faster-whisper` (CTranslate2) | Transcription to text with timestamps |
| YouTube URLs | `yt-dlp` + `youtube-transcript-api` | Auto-captions or manual subtitles; falls back to audio transcription via whisper |

### 3.2 Unified Document Schema

All parsers produce documents conforming to this internal schema:

```json
{
  "id": "unique_string",
  "text": "The extracted/transcribed content.",
  "metadata": {
    "source_type": "pdf | txt | md | docx | csv | html | jsonl | image | audio | youtube",
    "source_path": "./data/document.pdf",
    "title": "Optional document title",
    "any_other_key": "any_value"
  }
}
```

- `id` — required, auto-generated from file path + content hash if not provided
- `text` — required, extracted text content
- `metadata` — required but open schema; `source_type` and `source_path` are auto-populated by parsers

### 3.3 Config File

A domain is defined by a `domain.yaml` config file:

```yaml
domain:
  name: "specialty_coffee"
  description: "Expert knowledge on coffee brewing, extraction, and preparation"

corpus:
  sources:
    - path: "./data/documents/"         # directory of mixed files
      types: ["pdf", "docx", "txt"]     # filter by extension
      weight: 1.0
    - path: "./data/youtube_urls.txt"   # file containing YouTube URLs (one per line)
      type: "youtube"
      weight: 1.0
    - path: "./data/podcasts/"
      types: ["mp3", "wav"]
      transcription_model: "whisper-large-v3"
      weight: 1.0
    - path: "./data/images/"
      types: ["png", "jpg"]
      ocr_engine: "tesseract"           # or "claude-vision"
      weight: 0.8
    - path: "./data/legacy.jsonl"       # still supported
      weight: 1.0
  dedup_key: "id"

chunking:
  strategy: "semantic"          # fixed | sentence | semantic
  target_tokens: 200
  overlap_sentences: 1

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"

retrieval:
  strategy: "hybrid_rerank"     # dense | hybrid | hybrid_rerank
  top_k: 5
  rerank_candidates: 20
  reranker: "cross-encoder/ms-marco-MiniLM-L-6-v2"

llm:
  model: "claude-sonnet-4-20250514"     # generation model

eval:
  enabled: true
  qa_pairs_path: "./eval/coffee_qa.jsonl"
  judge: "claude-sonnet-4-20250514"     # fixed judge for consistency

benchmark:
  embedding_models:
    - "sentence-transformers/all-MiniLM-L6-v2"
    - "BAAI/bge-large-en-v1.5"
    - "nomic-ai/nomic-embed-text-v1.5"
    - "Alibaba-NLP/gte-large-en-v1.5"
    - "openai/text-embedding-3-small"
    - "openai/text-embedding-3-large"
    - "cohere/embed-english-v3.0"
    - "voyageai/voyage-3"
  llms:
    - "claude-sonnet-4-20250514"
    - "claude-haiku-4-20250514"
    - "gpt-4o"
    - "gpt-4o-mini"
    - "gemini-2.5-flash"
    - "gemini-2.5-pro"
    - "meta-llama/Llama-3.3-70B"
    - "mistral-large-latest"
  rerankers:
    - "cross-encoder/ms-marco-MiniLM-L-6-v2"
    - "BAAI/bge-reranker-v2-m3"
    - "cohere/rerank-english-v3.0"
    - "jinaai/jina-reranker-v2-base-multilingual"
  budget_limit_usd: 50.0               # cost cap for full benchmark run

api:
  host: "0.0.0.0"
  port: 8001
  metadata_filters:
    - "source"
    - "source_type"
    - "brew_method"
    - "topic_category"
```

### 3.4 Python API

For developers who want to integrate FluxRAG programmatically:

```python
from fluxrag import Pipeline

pipeline = Pipeline.from_config("domain.yaml")

# Ingest raw files into unified documents
pipeline.ingest()

# Build the knowledge base
pipeline.build()

# Run evals
results = pipeline.evaluate()
print(results.summary())

# Run model benchmarks
bench = pipeline.benchmark()
print(bench.summary())

# Query directly
answer = pipeline.query(
    question="Why does my espresso taste sour?",
    filters={"brew_method": "espresso"},
    top_k=5
)
```

---

## 4. Architecture

### 4.1 Module Map

```
fluxrag/
├── core/
│   ├── config.py          # Pydantic config loader and validator
│   ├── schema.py          # Input/output data models
│   └── pipeline.py        # Orchestrator — runs ingest/build/eval/benchmark/serve
├── ingestion/
│   ├── base.py            # Abstract parser interface
│   ├── router.py          # File type detection + dispatch to correct parser
│   ├── text.py            # TXT, Markdown parser
│   ├── pdf.py             # PDF parser (pymupdf + OCR fallback)
│   ├── docx.py            # DOCX parser (python-docx)
│   ├── csv_parser.py      # CSV/TSV parser (pandas)
│   ├── html.py            # HTML parser (bs4 + trafilatura)
│   ├── jsonl.py           # Legacy JSONL loader
│   ├── image.py           # Image OCR parser (tesseract / claude-vision)
│   ├── audio.py           # Audio transcription parser (faster-whisper)
│   ├── youtube.py         # YouTube transcript extractor (yt-dlp)
│   └── validator.py       # Schema validation and error reporting
├── chunking/
│   ├── base.py            # Abstract chunker interface
│   ├── fixed.py           # Strategy A: fixed-size with overlap
│   ├── sentence.py        # Strategy B: sentence-boundary aware (spaCy)
│   └── semantic.py        # Strategy C: semantic similarity grouping
├── embedding/
│   ├── base.py            # Abstract embedder interface
│   ├── local.py           # sentence-transformers wrapper (MiniLM, BGE, nomic, GTE)
│   ├── openai.py          # OpenAI embeddings API client
│   ├── cohere.py          # Cohere embeddings API client
│   ├── voyage.py          # Voyage AI embeddings API client
│   └── store.py           # ChromaDB / Qdrant vector store abstraction
├── retrieval/
│   ├── base.py            # Abstract retriever interface
│   ├── dense.py           # Strategy 1: dense vector search
│   └── hybrid.py          # Strategy 2: BM25 + dense via RRF
├── reranking/
│   ├── base.py            # Abstract reranker interface
│   ├── cross_encoder.py   # Local cross-encoder models (ms-marco, BGE)
│   ├── cohere.py          # Cohere rerank API
│   └── jina.py            # Jina reranker API
├── llm/
│   ├── base.py            # Abstract LLM interface
│   ├── anthropic.py       # Claude models (Sonnet 4, Haiku 4)
│   ├── openai.py          # GPT models (4o, 4o-mini)
│   ├── google.py          # Gemini models (2.5 Flash, 2.5 Pro)
│   ├── groq.py            # Groq-hosted models (Llama 3.3 70B)
│   └── mistral.py         # Mistral models (Large)
├── benchmark/
│   ├── runner.py          # Orchestrator: runs all benchmark matrices sequentially
│   ├── embedding_bench.py # Matrix B: embedding model comparison
│   ├── llm_bench.py       # Matrix D: LLM generation comparison
│   ├── reranker_bench.py  # Matrix C: reranker comparison
│   ├── cost_tracker.py    # Token counting + cost estimation per model
│   └── report.py          # Matrix report generator (markdown + JSON)
├── eval/
│   ├── generate_qa.py     # Synthetic QA generation (supports multiple LLMs)
│   ├── harness.py         # RAGAS evaluation runner
│   └── report.py          # Results table, markdown report generator
├── api/
│   ├── server.py          # FastAPI app
│   ├── routes.py          # /query, /health, /eval, /benchmark endpoints
│   └── models.py          # Request/response Pydantic models
├── cli/
│   └── main.py            # CLI: ingest | build | eval | benchmark | serve
└── examples/
    └── coffee/
        ├── domain.yaml    # Coffee domain config
        ├── data/          # Raw files (PDFs, YouTube URLs, audio, etc.)
        └── eval/          # QA test set + benchmark results
```

### 4.2 Data Flow

```
domain.yaml
    ↓
config.py        validates and loads all settings
    ↓
router.py        detects file types, dispatches to parsers
    ↓
parsers          PDF/DOCX/audio/image/YouTube/etc → unified Document objects
    ↓
chunker          splits text per chosen strategy
    ↓
encoder          embeds chunks via configured embedding model(s)
    ↓
store.py         persists vectors to ChromaDB/Qdrant
    ↓
retriever        serves queries via chosen retrieval + reranking strategy
    ↓
benchmark/       (optional) runs model comparison matrices
    ↓
eval/            RAGAS evaluation with configurable LLM judge
    ↓
api/server.py    exposes /query endpoint
```

---

## 5. Key Design Decisions & Tradeoffs

Every component choice in FluxRAG involved a tradeoff. These are documented here so the reasoning is auditable.

| Decision | Choice | Reason | Tradeoff |
|---|---|---|---|
| **Vector Store** | ChromaDB (dev) / Qdrant (prod) | ChromaDB is zero-config for local iteration; Qdrant supports filtering, sharding, and production loads | Two store backends to maintain. Mitigated by abstracting behind `store.py` interface |
| **Retrieval: Dense vs Hybrid** | Hybrid (BM25 + dense) as default | Dense search fails on exact terminology — grinder names, brew ratios, specific product models. BM25 catches these | Hybrid adds ~50ms latency and requires maintaining a separate BM25 index alongside the vector store |
| **Reranking** | Cross-encoder reranking enabled by default | +15–25 point precision improvement justifies the latency cost in a knowledge-quality-critical application | Adds ~350ms per query. Acceptable for async coaching tips, potentially too slow for real-time autocomplete |
| **Synthetic QA vs Human-Labelled** | Synthetic generation with manual review | No human-labelled coffee QA dataset exists. Synthetic generation scales to any domain — which is the point of FluxRAG being domain-agnostic | Risk of circular evaluation: LLM generates QA, LLM answers, LLM judges. Mitigated by manual review of all 100 pairs and using a fixed judge model |
| **Local vs API Embedding Models** | Benchmark both, recommend per budget | Local models (MiniLM, BGE) are free and fast but lower dimensional. API models (OpenAI, Cohere, Voyage) are higher quality but add cost and latency | The benchmark exists specifically to quantify this tradeoff per domain rather than guessing |
| **Chunking: Semantic over Fixed** | Semantic as recommended default | Coffee content shifts topics frequently within a single transcript — fixed chunking splits mid-explanation | Semantic chunking is 3–5× slower than fixed and depends on embedding quality for similarity thresholds |
| **LLM Judge** | Claude Sonnet 4 as fixed judge | Fixing the judge eliminates a variable during benchmarking. Sonnet 4 balances cost and evaluation quality | Potential bias toward Anthropic models in generation benchmarks. Matrix E (judge consistency) exists to validate this |
| **Sequential Benchmark Design** | A → B → C → D, each uses prior winner | Avoids combinatorial explosion (9 × 8 × 4 × 8 = 2,304 configs). Sequential design keeps the benchmark under $50 | Assumes independence between layers — e.g., that the best embedding doesn't change when you swap rerankers. Acceptable given budget constraints |

---

## 6. Chunking Strategies

All three strategies must be implemented. The benchmark compares them on RAGAS context recall using the same QA test set.

### Strategy A — Fixed-Size with Overlap (Baseline)
- Split by character count (~800 chars)
- 10% overlap between adjacent chunks
- No sentence boundary awareness
- Fast, simple, widely used — serves as the benchmark floor

### Strategy B — Sentence-Boundary Aware
- Use spaCy `en_core_web_sm` sentence tokeniser
- Group sentences until target token count (~200 tokens) is reached
- Never cut mid-sentence
- One-sentence overlap between chunks
- Expected improvement: +5–15 points context recall over Strategy A

### Strategy C — Semantic Chunking
- Embed each sentence with the configured embedding model
- Group consecutive sentences while cosine similarity stays above threshold (default 0.7)
- Split on topic shift (similarity drop)
- Variable chunk length, respects semantic boundaries
- Expected improvement: +10–25 points context recall over Strategy A on heterogeneous text

### Chunking Benchmark Output

Produced in `eval/matrix_a_report.md` as part of the 3×3 Matrix A:

| Strategy | Avg Chunk Tokens | Context Recall | Context Precision |
|---|---|---|---|
| A — Fixed | ~160 | baseline | baseline |
| B — Sentence | ~180 | +Δ | +Δ |
| C — Semantic | ~220 | +Δ | +Δ |

---

## 7. Retrieval Strategies

### Strategy 1 — Dense Vector Search (Baseline)
- Embed query with same model used for corpus
- Cosine similarity search over ChromaDB
- Return top-k chunks
- Fails on exact keyword matches (grinder model names, specific ratios, etc.)

### Strategy 2 — Hybrid Search (BM25 + Dense)
- Run BM25 keyword search in parallel using `rank_bm25`
- Run dense vector search in parallel
- Merge results using Reciprocal Rank Fusion (RRF)
- Catches exact-match queries that semantic search misses
- Expected improvement: +8–15 points on queries with specific terminology

### Strategy 3 — Hybrid Search + Cross-Encoder Reranking
- Run hybrid search to retrieve top-20 candidates
- Pass all 20 to the configured reranker model
- Reranker scores each candidate against the original query independently
- Return top-k by reranker score
- Highest quality, ~350ms additional latency per query
- Expected improvement: +15–25 points context precision over Strategy 1

### Retrieval Benchmark Output

Part of Matrix A in `eval/matrix_a_report.md`:

| Strategy | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Avg Latency |
|---|---|---|---|---|---|
| 1 — Dense | baseline | baseline | baseline | baseline | ~100ms |
| 2 — Hybrid | +Δ | +Δ | +Δ | +Δ | ~150ms |
| 3 — Hybrid + Rerank | +Δ | +Δ | +Δ | +Δ | ~500ms |

---

## 8. Evaluation Framework

### 8.1 Philosophy

Build the eval harness before optimising anything. The harness is what separates this project from a demo. It answers: "did this change help or hurt?" without guessing.

### 8.2 Synthetic QA Generation

No human-labelled dataset exists for most domains. Generate one synthetically:

```
generate_qa.py
  input:  unified Document objects (from any source type)
  method: sample 150 diverse chunks (stratified by source_type + metadata tags)
  prompt: for each chunk, the configured LLM generates:
            - 1 question whose answer is contained in the chunk
            - ground truth answer
            - difficulty label (easy / medium / hard)
  filter: manual review, discard ambiguous or trivial pairs
  output: eval/qa_pairs.jsonl
            {"question": "...", "ground_truth": "...", "source_chunk_id": "..."}
  target: 100 high-quality pairs minimum
```

### 8.3 RAGAS Metrics

| Metric | What It Measures | Baseline Target | Stretch Target |
|---|---|---|---|
| Context Recall | Fraction of ground truth covered by retrieved chunks | > 0.70 | > 0.85 |
| Context Precision | Fraction of retrieved chunks that are actually relevant | > 0.65 | > 0.80 |
| Faithfulness | LLM answer uses only retrieved context, no hallucination | > 0.80 | > 0.90 |
| Answer Relevancy | Generated answer addresses the question asked | > 0.75 | > 0.85 |

### 8.4 Benchmark Matrices

Benchmarks run sequentially: A → B → C → D. Each matrix uses the winner from the previous one as the fixed variable. This gives a clear optimisation path without combinatorial explosion.

**Matrix A: Chunking × Retrieval (9 configurations)**

Fixed embedding model (all-MiniLM-L6-v2), fixed reranker (ms-marco-MiniLM), fixed LLM (Claude Sonnet 4).

|  | Dense | Hybrid | Hybrid + Rerank |
|---|---|---|---|
| **Fixed** | A1 (double baseline) | A2 | A3 |
| **Sentence** | B1 | B2 | B3 |
| **Semantic** | C1 | C2 | **C3 (expected best)** |

Purpose: find best chunking + retrieval combination.

**Matrix B: Embedding Model Comparison (8 models)**

Fixed chunking + retrieval (winner from Matrix A), fixed reranker, fixed LLM.

| Model | Type | Dimensions | Context Recall | Context Precision | Faithfulness | Answer Relevancy | Latency (ms/query) | Cost ($/1M tokens) |
|---|---|---|---|---|---|---|---|---|
| all-MiniLM-L6-v2 | Local | 384 | baseline | baseline | baseline | baseline | ~15 | $0 |
| BGE-large-en-v1.5 | Local | 1024 | | | | | | $0 |
| nomic-embed-text-v1.5 | Local | 768 | | | | | | $0 |
| GTE-large-en-v1.5 | Local | 1024 | | | | | | $0 |
| OpenAI text-embedding-3-small | API | 1536 | | | | | | $0.02 |
| OpenAI text-embedding-3-large | API | 3072 | | | | | | $0.13 |
| Cohere embed-v3 | API | 1024 | | | | | | $0.10 |
| Voyage AI voyage-3 | API | 1024 | | | | | | $0.06 |

Purpose: find best embedding model for quality/cost/latency tradeoff.

**Matrix C: Reranker Comparison (4 models)**

Fixed chunking + retrieval + embedding (winners from A + B), fixed LLM.

| Reranker | Type | Context Precision | Answer Relevancy | Latency (ms/query) | Cost |
|---|---|---|---|---|---|
| ms-marco-MiniLM-L-6-v2 | Local | baseline | baseline | ~350 | $0 |
| BGE-reranker-v2-m3 | Local | | | | $0 |
| Cohere rerank-v3 | API | | | | $1/1K searches |
| Jina reranker-v2 | Local | | | | $0 |

Purpose: find best reranker.

**Matrix D: LLM Generation Comparison (8 models)**

All retrieval components fixed (winners from A + B + C). Same 100-pair QA set. Same judge (Claude Sonnet 4).

| LLM | Provider | Faithfulness | Answer Relevancy | Avg Latency | Cost (input/output per 1M tokens) |
|---|---|---|---|---|---|
| Claude Sonnet 4 | Anthropic | | | | $3 / $15 |
| Claude Haiku 4 | Anthropic | | | | $0.25 / $1.25 |
| GPT-4o | OpenAI | | | | $2.50 / $10 |
| GPT-4o-mini | OpenAI | | | | $0.15 / $0.60 |
| Gemini 2.5 Flash | Google | | | | $0.15 / $0.60 |
| Gemini 2.5 Pro | Google | | | | $1.25 / $10 |
| Llama 3.3 70B | Groq | | | | ~$0.59 / $0.79 |
| Mistral Large | Mistral | | | | $2 / $6 |

Purpose: find best generation LLM for quality/cost/latency. Produces a clear recommendation for enterprise budgets vs. startup budgets.

**Matrix E: LLM-as-Judge Consistency (stretch goal)**

| Judge LLM | Agreement with Claude Sonnet 4 (reference) | Self-Consistency | Cost |
|---|---|---|---|
| Claude Sonnet 4 | reference | | |
| GPT-4o | | | |
| Gemini 2.5 Pro | | | |

Purpose: validate that different judges produce consistent evaluation results.

---

## 9. Failure Modes & Limitations

No system works perfectly. These are the known failure modes, their impact, and how FluxRAG handles them.

| Failure Mode | Impact | Mitigation |
|---|---|---|
| **Semantic chunking splits on false topic boundaries** | Embedding similarity between adjacent sentences dips due to phrasing, not actual topic change. Produces fragmented chunks that lose context | Tunable similarity threshold (default 0.7). Floor on minimum chunk size prevents single-sentence fragments. Fixed chunking available as fallback |
| **Transcription noise from YouTube/audio sources** | Whisper and auto-captions introduce errors — misheard terminology, missing punctuation, speaker attribution loss. Degrades retrieval for exact terms | BM25 component in hybrid search partially compensates. Quality score in metadata allows filtering low-confidence transcripts at query time |
| **Reranker positional bias** | Cross-encoders can exhibit slight preference for candidates appearing earlier in the input list. Affects precision on borderline-relevant chunks | Shuffle candidate order before reranking. Benchmark Matrix C quantifies this per reranker model |
| **LLM judge inconsistency across runs** | RAGAS metrics depend on LLM judgment. Same query can score differently on repeated runs, adding noise to benchmark comparisons | Fix the judge model (Claude Sonnet 4) across all evaluations. Matrix E validates cross-judge agreement. Run each eval 3× and report mean ± std |
| **Noisy ingestion from heterogeneous sources** | PDFs with complex layouts, HTML with boilerplate leakage, OCR errors on low-resolution images. Junk chunks pollute the vector store | Parser-level validation in `validator.py`. Minimum text length threshold per chunk. Manual review of 50 random chunks per new source before merging |
| **Synthetic QA circularity** | LLM generates questions, LLM answers them, LLM judges the answers. Risk of inflated scores that don't reflect real user queries | Manual review of all QA pairs. Difficulty stratification (easy/medium/hard). Plan to validate against real user queries from coffeecoach.app once deployed |
| **Domain transfer assumptions** | Benchmark results on coffee domain may not generalize. Optimal chunking/retrieval strategy depends on content characteristics | FluxRAG is designed to re-run the full benchmark on any new domain. The pipeline is the product, not the coffee-specific results |

---

## 10. FastAPI Endpoint

### 10.1 Routes

```
POST /query
  body:  { "question": str, "filters": dict, "top_k": int }
  returns: { "answer": str, "sources": [{ "id", "text", "metadata", "score" }], "latency_ms": int }

GET  /health
  returns: { "status": "ok", "corpus_size": int, "index_ready": bool }

GET  /eval/latest
  returns: latest RAGAS scores from most recent eval run

GET  /benchmark/latest
  returns: latest model benchmark results across all matrices

POST /benchmark/run
  body:  { "matrices": ["embedding", "reranker", "llm"], "budget_limit_usd": 20.0 }
  returns: { "job_id": str, "status": "running" }

GET  /benchmark/status/{job_id}
  returns: progress and partial results
```

### 10.2 Latency Budget (Hetzner CX21)

| Step | Budget |
|---|---|
| Query embedding | 30ms |
| BM25 search | 30ms |
| Dense search | 50ms |
| Reranking | 350ms |
| LLM generation | 800ms |
| **Total** | **< 1300ms** |

### 10.3 Deployment

- Runs as a systemd service on Hetzner
- Sits behind existing Nginx + Certbot setup
- Exposed at `https://api.coffeecoach.app/rag/query`
- Containerised via Docker for portability

---

## 11. Coffee Domain Case Study

### 11.1 Corpus

Demonstrates FluxRAG's universal ingestion across multiple source types:

| Source | Input Type | Parser | Target Chunks |
|---|---|---|---|
| James Hoffmann (YouTube) | YouTube URL | youtube.py | ~1,700 |
| Sprometheus (YouTube) | YouTube URL | youtube.py | ~800 |
| Lance Hedrick (YouTube) | YouTube URL | youtube.py | ~800 |
| European Coffee Trip (YouTube) | YouTube URL | youtube.py | ~600 |
| Scott Rao (YouTube) | YouTube URL | youtube.py | ~400 |
| Scott Rao (Blog PDFs) | PDF | pdf.py | ~200 |
| SCA Research Papers | PDF | pdf.py | ~300 |
| Barista Hustle articles | HTML | html.py | ~500 |
| **Total** | | | **~5,300+** |

### 11.2 Metadata Enrichment

After corpus merge, run a batch enrichment pass using the configured LLM to tag each chunk:

```json
{
  "brew_method": "espresso | v60 | aeropress | chemex | moka | siphon | general",
  "topic_category": "extraction | grind | water | equipment | technique | origin | tasting",
  "difficulty": "beginner | intermediate | advanced"
}
```

These tags become available as query-time filters in the API.

### 11.3 Coffee Coach Integration

The running FastAPI endpoint powers two features in coffeecoach.app:

1. **Dynamic Coach's Secret tips** — RAG query on bean + brew method at session start, replaces hardcoded tips
2. **Troubleshooting flow** — user describes a problem ("my espresso is sour"), RAG retrieves relevant diagnostic content and generates a coaching response

---

## 12. Production Impact (CoffeeCoach)

### Before FluxRAG

CoffeeCoach v1 used a YouTube scraper ([youtube-rag-scraper](https://github.com/rav4nn/youtube-rag-scraper)) to pull transcripts from James Hoffmann's channel, embedded them with a single model, and served them through dense-only retrieval. Coaching tips were generated from a hardcoded decision tree — a static mapping of brew method + problem → advice. This worked for common cases but couldn't scale: adding new content meant re-running a brittle scraper, the decision tree required manual updates for every new scenario, and there was no way to measure whether retrieval quality was actually good.

### What Changed

| Aspect | Before (v1) | After (FluxRAG) |
|---|---|---|
| **Data sources** | Single YouTube channel (scraper) | 8+ sources across 5 format types (YouTube, PDF, HTML, audio, images) |
| **Ingestion** | Custom scraper, single format | Universal parser router — drop files in, get documents out |
| **Retrieval** | Dense-only, no evaluation | Hybrid + reranking, benchmarked across 3 strategies |
| **Coaching logic** | Hardcoded decision tree | RAG-generated responses from retrieved context |
| **Quality measurement** | None — manual spot-checking | RAGAS evaluation harness, 100-pair QA test set, 4 benchmark matrices |
| **Adding new content** | Re-scrape + manually update tree | Add files to `data/`, re-run `pipeline.ingest()` |
| **Model selection** | Single embedding + single LLM, chosen by intuition | Benchmarked 8 embeddings × 8 LLMs × 4 rerankers with documented tradeoffs |

### Why This Matters

The decision tree approach hit a ceiling: it could only answer questions someone had explicitly programmed. A user asking "why does my V60 stall halfway through drawdown?" would get a generic "adjust grind coarser" response — or nothing. With FluxRAG, the system retrieves specific context from multiple expert sources (Hoffmann explaining V60 flow dynamics, Rao on extraction physics, SCA research on particle distribution) and generates a contextual answer that addresses the actual problem.

The shift from hardcoded logic to evaluated RAG also made the system auditable. Every response now traces back to source chunks with relevance scores — the team can see *why* the system said what it said, not just *what* it said.

---

## 13. Delivery Phases

| Phase | Work | Duration | Done When |
|---|---|---|---|
| **0 — Scaffold** | Set up FluxRAG package structure. Define all abstract base interfaces for parsers, embedders, rerankers, LLMs. Pydantic config models. | 2 days | `fluxrag/` package importable, all ABCs defined |
| **1 — Ingestion Layer** | Implement file router + all 10 parsers (text, MD, PDF, DOCX, CSV, HTML, JSONL, image, audio, YouTube). Unit test each parser. | 1.5 weeks | All parsers pass tests with sample files |
| **2 — Baseline** | Ingest coffee corpus via new ingestion layer. Run RAGAS with Strategy A + Strategy 1 + all-MiniLM-L6-v2. Establish baseline. | 3 days | Baseline numbers in `eval/baseline_report.md` |
| **3 — Matrix A: Chunking × Retrieval** | Implement chunking strategies B/C and retrieval strategies 2/3. Run full 3×3 matrix. | 1.5 weeks | `eval/matrix_a_report.md` |
| **4 — Matrix B: Embedding Benchmark** | Implement embedding provider clients (local + OpenAI + Cohere + Voyage). Run across all 8 models. | 1 week | `eval/matrix_b_embedding_report.md` |
| **5 — Matrix C: Reranker Benchmark** | Implement reranker provider clients. Run across all 4 rerankers. | 4 days | `eval/matrix_c_reranker_report.md` |
| **6 — Matrix D: LLM Benchmark** | Implement LLM provider clients (Anthropic + OpenAI + Google + Groq + Mistral). Run across all 8 LLMs. | 1 week | `eval/matrix_d_llm_report.md` |
| **7 — Production Deploy** | Dockerise, deploy to Hetzner, wire to coffeecoach.app with winning config. | 1 week | Live endpoint at api.coffeecoach.app |
| **8 — Portfolio Wrap-up** | README with full methodology, all benchmark matrices, architecture diagram, cost analysis. | 3 days | Repo is self-explanatory |

---

## 14. Tech Stack

| Layer | Choice | Reason |
|---|---|---|
| **Config** | Pydantic v2 + PyYAML | Validated, typed config with good error messages |
| **PDF Parsing** | `pymupdf` (PyMuPDF) | Fast, handles most PDFs; OCR fallback via `pytesseract` |
| **DOCX Parsing** | `python-docx` | Standard DOCX extraction |
| **HTML Parsing** | `beautifulsoup4` + `trafilatura` | Boilerplate removal, main content extraction |
| **OCR** | `pytesseract` or Claude Vision API | Local or cloud OCR for images/scanned PDFs |
| **Audio Transcription** | `faster-whisper` (CTranslate2) | Faster than OpenAI whisper, runs on CPU |
| **YouTube** | `yt-dlp` + `youtube-transcript-api` | Transcript extraction; falls back to audio + whisper |
| **Encoding Detection** | `chardet` | Auto-detect text file encodings |
| **Chunking** | spaCy `en_core_web_sm` | Sentence tokenisation for Strategy B |
| **Embedding (local)** | `sentence-transformers` | all-MiniLM, BGE, nomic, GTE |
| **Embedding (API)** | `openai`, `cohere`, `voyageai` SDKs | Commercial embedding models |
| **Vector Store** | ChromaDB (dev), Qdrant (prod) | Local-first, no infra needed for dev |
| **BM25** | `rank_bm25` | Lightweight, no external service |
| **Reranking (local)** | `sentence-transformers` CrossEncoder | ms-marco, BGE-reranker |
| **Reranking (API)** | `cohere`, `requests` (Jina) | Commercial rerankers |
| **LLM — Anthropic** | `anthropic` SDK | Claude Sonnet 4 / Haiku 4 |
| **LLM — OpenAI** | `openai` SDK | GPT-4o / GPT-4o-mini |
| **LLM — Google** | `google-genai` SDK | Gemini 2.5 Flash / Pro |
| **LLM — Groq** | `groq` SDK | Llama 3.3 70B |
| **LLM — Mistral** | `mistralai` SDK | Mistral Large |
| **Evaluation** | RAGAS | Standard, LLM-judge-based RAG eval |
| **API** | FastAPI + Uvicorn | Existing stack in Coffee Coach |
| **Deployment** | Docker + systemd + Nginx | Existing Hetzner setup |

---

## 15. Repository Structure

```
flux-rag/
├── fluxrag/                # Python package
│   ├── core/
│   ├── ingestion/          # 12 files: base + router + 9 parsers + validator
│   ├── chunking/
│   ├── embedding/          # base + per-provider clients + store
│   ├── retrieval/
│   ├── reranking/          # base + per-provider clients
│   ├── llm/                # base + per-provider clients
│   ├── benchmark/          # runner + per-matrix benchmarks + cost tracker
│   ├── eval/
│   ├── api/
│   └── cli/
├── examples/
│   └── coffee/
│       ├── domain.yaml
│       ├── data/           # .gitignored, raw files of any type
│       └── eval/
│           ├── qa_pairs.jsonl
│           ├── baseline_report.md
│           ├── matrix_a_report.md
│           ├── matrix_b_embedding_report.md
│           ├── matrix_c_reranker_report.md
│           └── matrix_d_llm_report.md
├── tests/
│   ├── test_ingestion/
│   ├── test_chunking/
│   ├── test_embedding/
│   ├── test_reranking/
│   ├── test_llm/
│   └── test_benchmark/
├── CLAUDE.md
├── PRD.md
├── README.md
├── pyproject.toml
└── docker-compose.yml
```

---

## 16. CLAUDE.md Instructions (for Claude Code sessions)

When building this project with Claude Code, the `CLAUDE.md` file should contain:

```markdown
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
```

---

## 17. Success Criteria

### Technical
- Universal ingestion handles all 10 file types without manual preprocessing
- Strategy 3 (hybrid + rerank) beats Strategy 1 (dense) by ≥ 15 points on context recall
- All four RAGAS baseline targets met on the winning configuration
- Embedding benchmark produces a clear winner with at least 5-point improvement over baseline
- LLM benchmark identifies optimal cost/quality tradeoff for both enterprise and startup budgets
- Full benchmark suite runs within $50 budget cap
- FastAPI endpoint responds in < 1300ms on Hetzner CX21
- Pipeline runs end-to-end on a new domain from a single `domain.yaml` change

### Portfolio
- GitHub README tells a complete story: problem → methodology → results → live deployment
- All four benchmark matrices (A through D) are visible in the README with actual numbers
- Cost analysis table shows $/query for each model combination
- coffeecoach.app demonstrates the RAG layer in a live user flow
- Project can be explained end-to-end in a 10-minute technical interview

---

## 18. Risks

| Risk | Mitigation |
|---|---|
| Synthetic QA quality is poor | Manual review all 100 pairs. Regenerate any that are ambiguous or trivially easy. |
| RAGAS scores plateau across all strategies | Add metadata pre-filtering as a fourth variable to test. |
| Reranking latency exceeds budget on Hetzner | Quantise the cross-encoder with `optimum`. Profile before optimising. |
| Context drift across long Claude Code sessions | Follow CLAUDE.md strictly. One phase per session. Update tasks/todo.md at session end. |
| Corpus quality drops with additional sources | Review 50 random chunks from each new source before merging. |
| API costs exceed budget during benchmarking | `cost_tracker.py` enforces hard limits; run on subset first. |
| Audio transcription quality varies by accent/noise | Use whisper-large-v3; add quality score to metadata. |
| OCR fails on complex layouts | Fallback chain: pymupdf text → pytesseract → Claude Vision. |
| Commercial API rate limits during benchmark | Exponential backoff; stagger API calls across providers. |
| YouTube transcript unavailable | Fallback: download audio via yt-dlp, transcribe with whisper. |
| Model versions change during project | Pin exact model versions in config; log model version with all results. |

---

*This PRD is the single source of truth for FluxRAG. All Claude Code sessions should start by reading this file and tasks/todo.md.*
