"""Shared fixtures for ingestion tests."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory that's cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_txt(tmp_dir: Path) -> Path:
    """A plain text file with real content."""
    p = tmp_dir / "sample.txt"
    p.write_text(
        "Coffee extraction depends on grind size, water temperature, and brew time. "
        "A finer grind increases surface area, leading to faster extraction. "
        "Water temperature between 90-96°C is ideal for most brew methods.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def empty_txt(tmp_dir: Path) -> Path:
    p = tmp_dir / "empty.txt"
    p.write_text("", encoding="utf-8")
    return p


@pytest.fixture
def minimal_txt(tmp_dir: Path) -> Path:
    p = tmp_dir / "minimal.txt"
    p.write_text("Hello", encoding="utf-8")
    return p


@pytest.fixture
def sample_md(tmp_dir: Path) -> Path:
    p = tmp_dir / "guide.md"
    p.write_text(
        "# Brewing Guide\n\n"
        "## V60 Method\n\n"
        "Use a medium-fine grind. Start with a 30-second bloom phase using twice "
        "the weight of coffee in water. Then pour in slow, concentric circles.",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def sample_csv(tmp_dir: Path) -> Path:
    p = tmp_dir / "beans.csv"
    p.write_text(
        "name,origin,process\n"
        "Yirgacheffe,Ethiopia,Washed\n"
        "Gesha,Panama,Natural\n"
        "Blue Mountain,Jamaica,Washed\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def empty_csv(tmp_dir: Path) -> Path:
    p = tmp_dir / "empty.csv"
    p.write_text("name,origin,process\n", encoding="utf-8")
    return p


@pytest.fixture
def sample_html(tmp_dir: Path) -> Path:
    p = tmp_dir / "article.html"
    p.write_text(
        "<html><head><title>Espresso Guide</title></head>"
        "<body>"
        "<nav>Menu items here</nav>"
        "<article>"
        "<h1>Espresso Basics</h1>"
        "<p>Espresso is a concentrated coffee brewed by forcing hot water "
        "through finely ground coffee under pressure. A standard shot uses "
        "18 grams of coffee and yields about 36 grams of liquid in 25-30 seconds.</p>"
        "</article>"
        "<footer>Copyright 2026</footer>"
        "</body></html>",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def empty_html(tmp_dir: Path) -> Path:
    p = tmp_dir / "empty.html"
    p.write_text("<html><body></body></html>", encoding="utf-8")
    return p


@pytest.fixture
def sample_jsonl(tmp_dir: Path) -> Path:
    p = tmp_dir / "legacy.jsonl"
    lines = [
        json.dumps({"id": "doc_1", "text": "Grind size affects extraction rate significantly."}),
        json.dumps({"id": "doc_2", "text": "Water quality is the most overlooked variable in coffee brewing."}),
        "",  # empty line
        json.dumps({"id": "doc_3", "text": ""}),  # empty text, should be skipped
    ]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


@pytest.fixture
def empty_jsonl(tmp_dir: Path) -> Path:
    p = tmp_dir / "empty.jsonl"
    p.write_text("", encoding="utf-8")
    return p
