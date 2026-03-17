"""HTML parser using BeautifulSoup and trafilatura."""

from __future__ import annotations

from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class HTMLParser(AbstractParser):
    """Parses HTML files, extracting main content and removing boilerplate."""

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        raw_html = path.read_text(encoding="utf-8", errors="replace")

        text = self._extract_with_trafilatura(raw_html)
        if not text:
            text = self._extract_with_bs4(raw_html)

        if not text or not text.strip():
            return []

        title = self._extract_title(raw_html)

        return [
            Document(
                text=text,
                metadata={
                    "source_type": "html",
                    "source_path": str(path),
                    "title": title or path.stem,
                },
            )
        ]

    def _extract_with_trafilatura(self, html: str) -> str:
        """Primary extraction using trafilatura for boilerplate removal."""
        try:
            import trafilatura

            result = trafilatura.extract(html)
            return result or ""
        except Exception:
            return ""

    def _extract_with_bs4(self, html: str) -> str:
        """Fallback extraction using BeautifulSoup."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        return soup.get_text(separator="\n", strip=True)

    def _extract_title(self, html: str) -> str:
        """Extract the page title."""
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        title_tag = soup.find("title")
        return title_tag.get_text(strip=True) if title_tag else ""

    def supported_extensions(self) -> list[str]:
        return [".html", ".htm"]
