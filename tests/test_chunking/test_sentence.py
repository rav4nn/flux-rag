"""Tests for sentence-boundary chunking strategy."""

from fluxrag.chunking.sentence import SentenceChunker, _split_sentences
from fluxrag.core.schema import Document


chunker = SentenceChunker()


def _make_doc(text: str) -> Document:
    return Document(text=text, metadata={"source_type": "txt", "source_path": "test.txt"})


def test_splits_by_sentence_boundaries() -> None:
    text = (
        "Espresso requires a fine grind. The ideal pressure is 9 bars. "
        "Water temperature should be around 93 degrees Celsius. "
        "A standard shot takes 25-30 seconds to pull."
    )
    doc = _make_doc(text)
    chunks = chunker.chunk(doc, target_tokens=20)
    assert len(chunks) >= 2
    # No chunk should cut mid-sentence
    for chunk in chunks:
        assert chunk.text.rstrip().endswith((".", "!", "?")) or chunk == chunks[-1]


def test_single_sentence() -> None:
    doc = _make_doc("Just one sentence here.")
    chunks = chunker.chunk(doc, target_tokens=200)
    assert len(chunks) == 1
    assert chunks[0].text == "Just one sentence here."


def test_long_text_creates_multiple_chunks() -> None:
    sentences = [f"Sentence number {i} about coffee brewing." for i in range(20)]
    doc = _make_doc(" ".join(sentences))
    chunks = chunker.chunk(doc, target_tokens=30)
    assert len(chunks) > 1


def test_strategy_name() -> None:
    assert chunker.strategy_name == "sentence"


def test_sentence_splitter() -> None:
    text = "First sentence. Second sentence. Third sentence."
    sentences = _split_sentences(text)
    assert len(sentences) == 3
