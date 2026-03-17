"""YouTube transcript extractor using youtube-transcript-api with yt-dlp fallback."""

from __future__ import annotations

import logging
import re
import tempfile
from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser

logger = logging.getLogger(__name__)


class YouTubeParser(AbstractParser):
    """Extracts transcripts from YouTube URLs.

    Strategy:
    1. Try youtube-transcript-api for captions (fast, no download)
    2. Fall back to yt-dlp audio download + whisper transcription
    """

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        """Parse a file containing YouTube URLs (one per line) or a single URL."""
        path = Path(source_path)

        if path.exists() and path.is_file():
            urls = [
                line.strip()
                for line in path.read_text().splitlines()
                if line.strip() and not line.strip().startswith("#")
            ]
        else:
            urls = [source_path]

        documents: list[Document] = []
        for url in urls:
            video_id = self._extract_video_id(url)
            if not video_id:
                logger.warning("Could not extract video ID from: %s", url)
                continue

            text, title = self._get_transcript(video_id, url, **kwargs)
            if not text or not text.strip():
                logger.warning("No transcript for: %s", url)
                continue

            documents.append(
                Document(
                    text=text,
                    metadata={
                        "source_type": "youtube",
                        "source_path": url,
                        "title": title or video_id,
                        "video_id": video_id,
                    },
                )
            )

        return documents

    def _extract_video_id(self, url: str) -> str:
        """Extract YouTube video ID from various URL formats."""
        patterns = [
            r"(?:v=|/v/|youtu\.be/)([a-zA-Z0-9_-]{11})",
            r"^([a-zA-Z0-9_-]{11})$",
        ]
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return ""

    def _get_transcript(
        self, video_id: str, url: str, **kwargs: object
    ) -> tuple[str, str]:
        """Try captions first, then fall back to audio transcription."""
        text, title = self._try_captions(video_id)
        if text:
            return text, title

        logger.info("No captions for %s, falling back to audio transcription", video_id)
        return self._try_audio_fallback(url, **kwargs)

    def _try_captions(self, video_id: str) -> tuple[str, str]:
        """Attempt to get transcript via youtube-transcript-api."""
        try:
            from youtube_transcript_api import YouTubeTranscriptApi

            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            parts = [entry.text for entry in transcript.snippets]
            text = " ".join(parts)
            return text, ""
        except Exception as e:
            logger.debug("Caption fetch failed for %s: %s", video_id, e)
            return "", ""

    def _try_audio_fallback(self, url: str, **kwargs: object) -> tuple[str, str]:
        """Download audio via yt-dlp and transcribe with whisper."""
        try:
            import yt_dlp
            from faster_whisper import WhisperModel

            model_size = str(kwargs.get("transcription_model", "base"))

            with tempfile.TemporaryDirectory() as tmpdir:
                output_path = str(Path(tmpdir) / "audio.%(ext)s")
                ydl_opts = {
                    "format": "bestaudio/best",
                    "outtmpl": output_path,
                    "quiet": True,
                    "no_warnings": True,
                    "postprocessors": [{
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": "wav",
                    }],
                }

                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = info.get("title", "") if info else ""

                audio_file = next(Path(tmpdir).glob("audio.*"), None)
                if not audio_file:
                    return "", title

                model = WhisperModel(model_size, device="cpu", compute_type="int8")
                segments, _ = model.transcribe(str(audio_file))
                text = " ".join(seg.text.strip() for seg in segments)
                return text, title

        except Exception as e:
            logger.warning("Audio fallback failed for %s: %s", url, e)
            return "", ""

    def supported_extensions(self) -> list[str]:
        return [".youtube"]
