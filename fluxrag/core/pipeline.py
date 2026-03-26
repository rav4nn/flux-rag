"""Pipeline orchestrator — ties all FluxRAG components together."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from fluxrag.core.config import FluxRAGConfig
from fluxrag.core.schema import (
    BenchmarkResult,
    Chunk,
    Document,
    EmbeddedChunk,
    EvalResult,
    QueryResult,
    RetrievalResult,
)

logger = logging.getLogger(__name__)


class Pipeline:
    """Main orchestrator for the FluxRAG pipeline.

    Usage:
        pipeline = Pipeline.from_config("domain.yaml")
        pipeline.ingest()
        pipeline.build()
        results = pipeline.evaluate()
    """

    def __init__(self, config: FluxRAGConfig) -> None:
        self.config = config
        self._config_dir: Path = Path.cwd()
        self._documents: list[Document] = []
        self._chunks: list[Chunk] = []
        self._embedder = None
        self._store = None
        self._retriever = None
        self._generator = None

    @property
    def is_ready(self) -> bool:
        """Whether the pipeline is built and ready to answer queries."""
        return self._retriever is not None and self._embedder is not None

    @property
    def corpus_size(self) -> int:
        """Number of chunks in the vector store."""
        if self._store is None:
            return 0
        try:
            return self._store.count()
        except Exception:
            return 0

    @classmethod
    def from_config(cls, config_path: str) -> Pipeline:
        """Load a pipeline from a domain.yaml config file."""
        config = FluxRAGConfig.from_yaml(config_path)
        pipeline = cls(config)
        pipeline._config_dir = Path(config_path).parent.resolve()
        return pipeline

    def ingest(self) -> list[Document]:
        """Parse all configured sources into unified Document objects."""
        from fluxrag.ingestion.router import create_default_router
        from fluxrag.ingestion.validator import DocumentValidator

        router = create_default_router()
        validator = DocumentValidator()
        all_docs: list[Document] = []

        for source in self.config.corpus.sources:
            source_path = (self._config_dir / source.path).resolve()
            kwargs: dict[str, object] = {}
            if source.transcription_model:
                kwargs["transcription_model"] = source.transcription_model
            if source.ocr_engine:
                kwargs["ocr_engine"] = source.ocr_engine

            if source.type == "youtube_scraper":
                # youtube-rag-scraper JSON output
                from fluxrag.ingestion.youtube_scraper import YouTubeScraperParser

                parser = YouTubeScraperParser()
                docs = parser.parse(str(source_path), **kwargs)
                all_docs.extend(docs)
                logger.info("Ingested %d transcripts from scraper file: %s", len(docs), source_path)

            elif source.type == "youtube":
                # YouTube URL file — pass directly to youtube parser
                from fluxrag.ingestion.youtube import YouTubeParser

                parser = YouTubeParser()
                docs = parser.parse(str(source_path), **kwargs)
                all_docs.extend(docs)
                logger.info("Ingested %d documents from YouTube: %s", len(docs), source_path)

            elif source_path.is_dir():
                extensions = {f".{t}" for t in (source.types or [])}
                for file_path in sorted(source_path.rglob("*")):
                    if not file_path.is_file():
                        continue
                    if extensions and file_path.suffix.lower() not in extensions:
                        continue
                    try:
                        docs = router.parse(str(file_path), **kwargs)
                        all_docs.extend(docs)
                    except (ValueError, Exception) as e:
                        logger.warning("Failed to parse %s: %s", file_path, e)

            elif source_path.is_file():
                try:
                    docs = router.parse(str(source_path), **kwargs)
                    all_docs.extend(docs)
                except (ValueError, Exception) as e:
                    logger.warning("Failed to parse %s: %s", source_path, e)
            else:
                logger.warning("Source path does not exist: %s", source_path)

        self._documents = validator.validate_batch(all_docs)
        logger.info("Ingested %d valid documents from %d sources",
                     len(self._documents), len(self.config.corpus.sources))
        return self._documents

    def build(self) -> None:
        """Chunk, embed, and store all documents in the vector store."""
        if not self._documents:
            raise RuntimeError("No documents to build from. Run ingest() first.")

        chunker = self._create_chunker()
        self._embedder = self._create_embedder()
        self._store = self._create_store()

        # Chunk all documents
        self._chunks = []
        for doc in self._documents:
            doc_chunks = chunker.chunk(doc, target_tokens=self.config.chunking.target_tokens)
            self._chunks.extend(doc_chunks)
        logger.info("Created %d chunks from %d documents", len(self._chunks), len(self._documents))

        # Embed and store in batches
        batch_size = 64
        for i in range(0, len(self._chunks), batch_size):
            batch = self._chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts)
            embedded = [
                EmbeddedChunk(chunk=c, embedding=e) for c, e in zip(batch, embeddings)
            ]
            self._store.add(embedded)

        logger.info("Stored %d embedded chunks in vector store", self._store.count())

        # Cache chunks to disk so load() can restore the pipeline without re-embedding
        import json as _json
        cache_dir = self._config_dir / ".fluxrag"
        cache_dir.mkdir(exist_ok=True)
        with open(cache_dir / "chunks.json", "w", encoding="utf-8") as _f:
            _json.dump([c.model_dump() for c in self._chunks], _f)

        # Set up retriever
        self._retriever = self._create_retriever()

    def load(self) -> None:
        """Reconnect to a previously built index without re-embedding."""
        import json as _json

        cache_dir = self._config_dir / ".fluxrag"
        chunks_path = cache_dir / "chunks.json"
        if not chunks_path.exists():
            raise RuntimeError(
                f"No built index found at {cache_dir}. Run 'fluxrag build' first."
            )
        with open(chunks_path, encoding="utf-8") as _f:
            self._chunks = [Chunk(**c) for c in _json.load(_f)]
        self._embedder = self._create_embedder()
        self._store = self._create_store()
        self._retriever = self._create_retriever()
        logger.info(
            "Loaded pipeline: %d chunks, store has %d vectors",
            len(self._chunks),
            self._store.count(),
        )

    def evaluate(self) -> EvalResult:
        """Run evaluation against the QA test set."""
        if self._retriever is None:
            raise RuntimeError("Pipeline not built. Run build() first.")

        from fluxrag.eval.generate_qa import load_qa_pairs
        from fluxrag.eval.harness import EvalHarness

        qa_pairs_path = str((self._config_dir / self.config.eval.qa_pairs_path).resolve())
        qa_pairs = load_qa_pairs(qa_pairs_path)
        logger.info("Loaded %d QA pairs from %s", len(qa_pairs), self.config.eval.qa_pairs_path)

        generator = self._create_generator()
        judge = self._create_judge()

        harness = EvalHarness(
            retriever=self._retriever,
            generator=generator,
            judge=judge,
            top_k=self.config.retrieval.top_k,
        )

        return harness.evaluate(qa_pairs)

    def benchmark(self) -> BenchmarkResult:
        """Run Matrix A: Chunking × Retrieval benchmark."""
        if not self._documents:
            raise RuntimeError("No documents. Run ingest() first.")

        from fluxrag.benchmark.runner import MatrixARunner

        runner = MatrixARunner(
            documents=self._documents,
            embedder=self._create_embedder(),
            generator=self._create_generator(),
            judge=self._create_judge(),
            qa_pairs_path=self.config.eval.qa_pairs_path,
            reranker_model=self.config.retrieval.reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2",
            target_tokens=self.config.chunking.target_tokens,
            top_k=self.config.retrieval.top_k,
            rerank_candidates=self.config.retrieval.rerank_candidates,
        )
        return runner.run()

    def query(
        self,
        question: str,
        filters: dict[str, Any] | None = None,
        top_k: int = 5,
    ) -> QueryResult:
        """Query the knowledge base."""
        if self._retriever is None or self._embedder is None:
            raise RuntimeError("Pipeline not built. Run build() first.")

        start = time.time()

        results = self._retriever.retrieve(question, top_k=top_k, filters=filters)
        contexts = [r.chunk.text for r in results]

        generator = self._create_generator()
        answer = generator.generate(question, context=contexts)

        latency_ms = (time.time() - start) * 1000

        return QueryResult(
            answer=answer,
            sources=results,
            latency_ms=latency_ms,
        )

    def serve(self, config_path: str = "") -> None:
        """Start the FastAPI server."""
        from fluxrag.api.server import run_server

        run_server(
            config_path=config_path,
            host=self.config.api.host,
            port=self.config.api.port,
        )

    # --- Factory methods ---

    def _create_chunker(self):
        strategy = self.config.chunking.strategy
        if strategy == "fixed":
            from fluxrag.chunking.fixed import FixedChunker
            return FixedChunker()
        elif strategy == "sentence":
            from fluxrag.chunking.sentence import SentenceChunker
            return SentenceChunker()
        elif strategy == "semantic":
            from fluxrag.chunking.semantic import SemanticChunker
            return SemanticChunker(self._create_embedder())
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    def _create_embedder(self):
        from fluxrag.embedding.factory import create_embedder

        return create_embedder(self.config.embedding.model)

    def _create_store(self):
        from fluxrag.embedding.chromadb_store import ChromaDBStore

        persist_dir = str(self._config_dir / ".fluxrag" / "chroma")
        return ChromaDBStore(collection_name=self.config.domain.name, persist_directory=persist_dir)

    def _create_retriever(self):
        strategy = self.config.retrieval.strategy
        if strategy == "dense":
            from fluxrag.retrieval.dense import DenseRetriever
            return DenseRetriever(self._embedder, self._store)
        elif strategy == "hybrid":
            from fluxrag.retrieval.hybrid import HybridRetriever
            return HybridRetriever(self._embedder, self._store, self._chunks)
        elif strategy == "hybrid_rerank":
            from fluxrag.retrieval.hybrid import HybridRetriever
            from fluxrag.retrieval.hybrid_rerank import HybridRerankRetriever
            from fluxrag.reranking.factory import create_reranker
            hybrid = HybridRetriever(self._embedder, self._store, self._chunks)
            reranker = create_reranker(
                self.config.retrieval.reranker or "cross-encoder/ms-marco-MiniLM-L-6-v2"
            )
            return HybridRerankRetriever(
                hybrid, reranker, self.config.retrieval.rerank_candidates
            )
        raise ValueError(f"Unknown retrieval strategy: {strategy}")

    def _create_generator(self):
        if self._generator is None:
            from fluxrag.llm.factory import create_llm

            self._generator = create_llm(self.config.llm.model)
        return self._generator

    def _create_judge(self):
        from fluxrag.llm.factory import create_llm

        return create_llm(self.config.eval.judge)
