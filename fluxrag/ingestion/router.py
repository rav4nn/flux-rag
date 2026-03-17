"""File type detection and dispatch to the appropriate parser."""

from __future__ import annotations

from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class ParserRouter:
    """Detects file types and dispatches to the appropriate parser."""

    def __init__(self) -> None:
        self._parsers: dict[str, AbstractParser] = {}

    def register(self, parser: AbstractParser) -> None:
        """Register a parser for its supported extensions."""
        for ext in parser.supported_extensions():
            self._parsers[ext.lower()] = parser

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        """Detect file type and parse using the appropriate parser."""
        ext = Path(source_path).suffix.lower()
        parser = self._parsers.get(ext)
        if parser is None:
            raise ValueError(f"No parser registered for extension: {ext}")
        return parser.parse(source_path, **kwargs)

    @property
    def supported_extensions(self) -> list[str]:
        """Return all registered extensions."""
        return sorted(self._parsers.keys())
