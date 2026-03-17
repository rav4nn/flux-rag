"""Schema validation for parsed documents."""

from __future__ import annotations

import logging

from fluxrag.core.schema import Document

logger = logging.getLogger(__name__)


class DocumentValidator:
    """Validates Document objects against the unified schema."""

    MIN_TEXT_LENGTH = 10

    def validate(self, document: Document) -> Document:
        """Validate and normalize a document. Raises ValueError on invalid documents."""
        if len(document.text.strip()) < self.MIN_TEXT_LENGTH:
            raise ValueError(
                f"Document {document.id} text too short ({len(document.text.strip())} chars)"
            )
        if "source_type" not in document.metadata:
            raise ValueError(f"Document {document.id} missing required metadata key: source_type")
        return document

    def validate_batch(self, documents: list[Document]) -> list[Document]:
        """Validate a batch, returning only valid documents. Logs warnings for invalid ones."""
        valid = []
        for doc in documents:
            try:
                valid.append(self.validate(doc))
            except ValueError as e:
                logger.warning("Skipping invalid document: %s", e)
        return valid
