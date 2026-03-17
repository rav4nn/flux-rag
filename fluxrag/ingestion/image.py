"""Image OCR parser using pytesseract."""

from __future__ import annotations

from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class ImageParser(AbstractParser):
    """Parses images via OCR to extract text content."""

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)

        text = self._ocr(path)
        if not text or not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata={
                    "source_type": "image",
                    "source_path": str(path),
                    "title": path.stem,
                },
            )
        ]

    def _ocr(self, path: Path) -> str:
        """Run OCR on an image file."""
        import pytesseract
        from PIL import Image

        img = Image.open(path)
        return pytesseract.image_to_string(img).strip()

    def supported_extensions(self) -> list[str]:
        return [".png", ".jpg", ".jpeg", ".tiff", ".tif"]
