"""End-to-end integration test with a toy corpus (no API keys needed)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from fluxrag.chunking.fixed import FixedChunker
from fluxrag.core.schema import Chunk, Document, EmbeddedChunk
from fluxrag.embedding.chromadb_store import ChromaDBStore
from fluxrag.embedding.local import LocalEmbedder
from fluxrag.retrieval.dense import DenseRetriever


# Toy coffee corpus — 10 documents that should produce ~10+ chunks
TOY_CORPUS = [
    "Espresso is brewed by forcing hot water through finely ground coffee at 9 bars of pressure. A standard double shot uses 18 grams of coffee and yields approximately 36 grams of liquid in 25-30 seconds.",
    "The V60 pour over method uses a conical dripper with spiral ribs. Start with a 30-second bloom phase using twice the weight of coffee in water, then pour in slow concentric circles.",
    "French press uses immersion brewing with a coarse grind. Add coffee, pour water at 96°C, steep for 4 minutes, then press the plunger slowly to separate grounds from liquid.",
    "Grind size is the single most impactful variable in coffee extraction. Finer grinds increase surface area, leading to faster extraction. Too fine leads to over-extraction and bitterness.",
    "Water temperature affects extraction rate exponentially. The ideal range is 90-96°C for most brew methods. Light roasts benefit from higher temperatures, dark roasts from lower.",
    "The Specialty Coffee Association defines ideal extraction yield as 18-22% of the coffee's soluble mass. Under-extraction produces sour, acidic flavors. Over-extraction produces bitterness.",
    "Coffee freshness degrades rapidly after roasting. Peak flavor window is typically 7-21 days post-roast. Grinding accelerates staling — grind immediately before brewing for best results.",
    "Water quality accounts for over 98% of brewed coffee by weight. Hard water with high mineral content can mute flavors. The SCA recommends 75-250 mg/L total dissolved solids.",
    "Channeling in espresso occurs when water finds paths of least resistance through the coffee puck. This causes uneven extraction — some areas over-extracted, others under-extracted.",
    "The AeroPress is a versatile brewer that supports both immersion and pressure methods. Its portability and forgiving nature make it popular for travel and experimentation.",
]


@pytest.fixture(scope="module")
def embedder() -> LocalEmbedder:
    return LocalEmbedder("sentence-transformers/all-MiniLM-L6-v2")


@pytest.fixture
def built_retriever(embedder: LocalEmbedder) -> DenseRetriever:
    """Build a complete retrieval pipeline from the toy corpus."""
    store = ChromaDBStore(collection_name="test_e2e")
    store.reset()
    chunker = FixedChunker()

    all_chunks: list[EmbeddedChunk] = []
    for i, text in enumerate(TOY_CORPUS):
        doc = Document(text=text, metadata={"source_type": "txt", "source_path": f"toy_{i}.txt"})
        chunks = chunker.chunk(doc, chunk_size=400)
        for chunk in chunks:
            embedding = embedder.embed([chunk.text])[0]
            all_chunks.append(EmbeddedChunk(chunk=chunk, embedding=embedding))

    store.add(all_chunks)
    return DenseRetriever(embedder, store)


def test_toy_corpus_produces_chunks(embedder: LocalEmbedder) -> None:
    """Verify the toy corpus produces a reasonable number of chunks."""
    chunker = FixedChunker()
    total_chunks = 0
    for i, text in enumerate(TOY_CORPUS):
        doc = Document(text=text, metadata={"source_type": "txt", "source_path": f"toy_{i}.txt"})
        chunks = chunker.chunk(doc, chunk_size=400)
        total_chunks += len(chunks)
    assert total_chunks >= 10


def test_espresso_query_retrieves_espresso(built_retriever: DenseRetriever) -> None:
    results = built_retriever.retrieve("How do I brew espresso?", top_k=3)
    texts = " ".join(r.chunk.text.lower() for r in results)
    assert "espresso" in texts


def test_grind_query_retrieves_grind(built_retriever: DenseRetriever) -> None:
    results = built_retriever.retrieve("How does grind size affect coffee?", top_k=3)
    texts = " ".join(r.chunk.text.lower() for r in results)
    assert "grind" in texts


def test_water_query_retrieves_water(built_retriever: DenseRetriever) -> None:
    results = built_retriever.retrieve("What water temperature should I use?", top_k=3)
    texts = " ".join(r.chunk.text.lower() for r in results)
    assert "temperature" in texts or "water" in texts


def test_retrieval_scores_are_ordered(built_retriever: DenseRetriever) -> None:
    results = built_retriever.retrieve("coffee extraction", top_k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_ingest_from_directory(embedder: LocalEmbedder) -> None:
    """Test ingesting from a directory of text files."""
    from fluxrag.ingestion.router import create_default_router

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, text in enumerate(TOY_CORPUS[:3]):
            (Path(tmpdir) / f"doc_{i}.txt").write_text(text, encoding="utf-8")

        router = create_default_router()
        docs: list[Document] = []
        for f in sorted(Path(tmpdir).glob("*.txt")):
            docs.extend(router.parse(str(f)))

        assert len(docs) == 3
        assert all(d.metadata["source_type"] == "txt" for d in docs)
