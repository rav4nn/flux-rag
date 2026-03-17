"""Audio transcription parser using faster-whisper."""

from __future__ import annotations

import logging
from pathlib import Path

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser

logger = logging.getLogger(__name__)


class AudioParser(AbstractParser):
    """Transcribes audio files to text using faster-whisper."""

    DEFAULT_MODEL = "base"

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        model_size = str(kwargs.get("transcription_model", self.DEFAULT_MODEL))

        text, duration = self._transcribe(str(path), model_size)
        if not text or not text.strip():
            return []

        return [
            Document(
                text=text,
                metadata={
                    "source_type": "audio",
                    "source_path": str(path),
                    "title": path.stem,
                    "transcription_model": model_size,
                    "duration_seconds": duration,
                },
            )
        ]

    def _transcribe(self, audio_path: str, model_size: str) -> tuple[str, float]:
        """Transcribe audio and return (text, duration_seconds)."""
        from faster_whisper import WhisperModel

        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path)

        parts: list[str] = []
        for segment in segments:
            parts.append(segment.text.strip())

        return " ".join(parts), info.duration

    def supported_extensions(self) -> list[str]:
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
