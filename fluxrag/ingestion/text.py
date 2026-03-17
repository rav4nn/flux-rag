"""TXT and Markdown parser."""

from __future__ import annotations

from pathlib import Path

import chardet

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class TextParser(AbstractParser):
    """Parses plain text and Markdown files."""

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        raw = path.read_bytes()

        detection = chardet.detect(raw)
        encoding = detection.get("encoding") or "utf-8"
        text = raw.decode(encoding, errors="replace")

        if not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata={
                    "source_type": path.suffix.lstrip("."),
                    "source_path": str(path),
                    "title": path.stem,
                },
            )
        ]

    def supported_extensions(self) -> list[str]:
        return [".txt", ".md"]
