"""PDF parser using PyMuPDF with OCR fallback."""

from __future__ import annotations

from pathlib import Path

import pymupdf

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class PDFParser(AbstractParser):
    """Parses PDF files. Falls back to OCR for scanned pages."""

    MIN_TEXT_LENGTH = 50

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        doc = pymupdf.open(str(path))
        pages: list[str] = []

        for page in doc:
            text = page.get_text()
            if len(text.strip()) >= self.MIN_TEXT_LENGTH:
                pages.append(text.strip())
            else:
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    pages.append(ocr_text)

        doc.close()

        full_text = "\n\n".join(pages)
        if not full_text.strip():
            return []

        return [
            Document(
                text=full_text,
                metadata={
                    "source_type": "pdf",
                    "source_path": str(path),
                    "title": path.stem,
                    "page_count": len(pages),
                },
            )
        ]

    def _ocr_page(self, page: pymupdf.Page) -> str:
        """Attempt OCR on a page with minimal text."""
        try:
            import pytesseract
            from PIL import Image
            import io

            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            text = pytesseract.image_to_string(img)
            return text.strip() if len(text.strip()) >= self.MIN_TEXT_LENGTH else ""
        except Exception:
            return ""

    def supported_extensions(self) -> list[str]:
        return [".pdf"]
