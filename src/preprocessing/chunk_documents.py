import re

from src.ingestion.models import Document, DocumentChunk


DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200
BOUNDARY_WINDOW = 200


def _find_best_boundary(text: str, target: int) -> int:
    """
    Find the best natural boundary near the target position.

    Priority:
    1. Paragraph boundary
    2. Sentence boundary
    3. Hard boundary at target
    """

    if target >= len(text):
        return len(text)

    window_start = max(0, target - BOUNDARY_WINDOW)
    window_end = min(len(text), target + BOUNDARY_WINDOW)

    window = text[window_start:window_end]

    # Prefer paragraph boundaries.
    paragraph_matches = list(re.finditer(r"\n\s*\n", window))

    if paragraph_matches:
        nearest = min(
            paragraph_matches,
            key=lambda match: abs(
                (window_start + match.end()) - target
            ),
        )

        return window_start + nearest.end()

    # Fall back to sentence boundaries.
    sentence_matches = list(
        re.finditer(r"[.!?](?:\s+|$)", window)
    )

    if sentence_matches:
        nearest = min(
            sentence_matches,
            key=lambda match: abs(
                (window_start + match.end()) - target
            ),
        )

        return window_start + nearest.end()

    # Last resort: hard boundary.
    return target


def _chunk_text(
    text: str,
    chunk_size: int,
    overlap: int,
) -> list[str]:

    if not text.strip():
        return []

    chunks = []
    start = 0

    while start < len(text):

        target_end = min(
            start + chunk_size,
            len(text),
        )

        if target_end == len(text):
            end = len(text)
        else:
            end = _find_best_boundary(text, target_end)

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        next_start = max(
            0,
            end - overlap,
        )

        # Safety guard against getting stuck.
        if next_start <= start:
            next_start = end

        start = next_start

    return chunks


def chunk_document(
    document: Document,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[DocumentChunk]:
    """
    Split a Document into page-aware, naturally bounded chunks.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks: list[DocumentChunk] = []
    chunk_index = 0

    for section in document.sections:

        section_chunks = _chunk_text(
            text=section.text,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        for chunk_text in section_chunks:

            chunks.append(
                DocumentChunk(
                    chunk_id=f"{document.document_id}_{chunk_index}",
                    document_id=document.document_id,
                    filename=document.filename,
                    text=chunk_text,
                    source=document.source,
                    section=section.heading,
                    page=section.page,
                    chunk_index=chunk_index,
                )
            )

            chunk_index += 1

    return chunks