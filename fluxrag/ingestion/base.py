"""Abstract parser interface. All file parsers implement this."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fluxrag.core.schema import Document


class AbstractParser(ABC):
    """Base interface for all file parsers.

    Each parser converts a specific file type (PDF, audio, YouTube, etc.)
    into unified Document objects.
    """

    @abstractmethod
    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        """Parse a file or URL into unified Document objects.

        Args:
            source_path: Path to file or URL to parse.
            **kwargs: Parser-specific options (e.g., ocr_engine, transcription_model).

        Returns:
            List of Document objects with populated id, text, and metadata.
        """
        ...

    @abstractmethod
    def supported_extensions(self) -> list[str]:
        """Return list of file extensions this parser handles (e.g., ['.pdf'])."""
        ...
