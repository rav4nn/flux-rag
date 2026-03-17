"""CSV/TSV parser using pandas."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fluxrag.core.schema import Document
from fluxrag.ingestion.base import AbstractParser


class CSVParser(AbstractParser):
    """Parses CSV/TSV files. Each row group becomes a document."""

    DEFAULT_ROWS_PER_DOC = 10

    def parse(self, source_path: str, **kwargs: object) -> list[Document]:
        path = Path(source_path)
        sep = "\t" if path.suffix == ".tsv" else ","
        rows_per_doc = int(kwargs.get("rows_per_doc", self.DEFAULT_ROWS_PER_DOC))

        df = pd.read_csv(str(path), sep=sep)
        if df.empty:
            return []

        documents: list[Document] = []
        for start in range(0, len(df), rows_per_doc):
            chunk = df.iloc[start : start + rows_per_doc]
            text = chunk.to_string(index=False)
            if text.strip():
                documents.append(
                    Document(
                        text=text,
                        metadata={
                            "source_type": path.suffix.lstrip("."),
                            "source_path": str(path),
                            "title": path.stem,
                            "row_range": f"{start}-{start + len(chunk) - 1}",
                        },
                    )
                )

        return documents

    def supported_extensions(self) -> list[str]:
        return [".csv", ".tsv"]
