"""Parser for youtube-rag-scraper JSON output files.

Reads the JSON format produced by https://github.com/rav4nn/youtube-rag-scraper:
{
  "source_target": "CHANNEL_ID",
  "total_videos": 397,
  "videos": [
    {"id": "...", "title": "...", "transcript": "...", "channel_title": "...", ...}
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class YouTubeScraperParser(AbstractParser):
    """Loads transcripts from youtube-rag-scraper JSON output."""

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        documents: list[Document] = []

        with open(path, encoding="utf-8-sig") as f:
            data = json.load(f)

        videos = data.get("videos", [])
        for video in videos:
            transcript = (video.get("transcript") or "").strip()
            if not transcript or video.get("transcript_error"):
                continue

            documents.append(
                Document(
                    id=video.get("id", ""),
                    text=transcript,
                    metadata={
                        "title": video.get("title", ""),
                        "url": video.get("url", ""),
                        "channel": video.get("channel_title", ""),
                        "source_type": "youtube_transcript",
                        "duration": video.get("duration", ""),
                        "view_count": video.get("view_count", 0),
                        "source_path": str(path),
                    },
                )
            )

        return documents

    def supported_extensions(self) -> list[str]:
        # Don't register for .json globally — invoked explicitly via type: "youtube_scraper"
        return []
