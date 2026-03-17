"""Legacy JSONL loader for pre-formatted documents."""

from __future__ import annotations

import json
from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class JSONLParser(AbstractParser):
    """Loads documents from JSONL files (legacy format support)."""

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        documents: list[Document] = []

        with open(path, encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                text = data.get("text", "")
                if not text.strip():
                    continue

                metadata = data.get("metadata", {})
                metadata.setdefault("source_type", "jsonl")
                metadata.setdefault("source_path", str(path))

                documents.append(
                    Document(
                        id=data.get("id", ""),
                        text=text,
                        metadata=metadata,
                    )
                )

        return documents

    def supported_extensions(self) -> list[str]:
        return [".jsonl"]
