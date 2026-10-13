from __future__ import annotations

from typing import Sequence

import numpy as np

from src.ingestion.models import DocumentChunk


def store_chunks(
    collection,
    chunks: Sequence[DocumentChunk],
    embeddings: np.ndarray,
) -> None:
    """
    Store document chunks and their embeddings in a Chroma collection.
    """

    if len(chunks) != len(embeddings):
        raise ValueError(
            "Number of chunks must match number of embeddings"
        )

    if len(chunks) == 0:
        return

    collection.add(
        ids=[chunk.chunk_id for chunk in chunks],
        embeddings=embeddings.tolist(),
        documents=[chunk.text for chunk in chunks],
        metadatas=[
            {
                key: value 
                for key , value in {
                    
                    "document_id": chunk.document_id,
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "filename": chunk.filename,
                    "page": chunk.page,
                    "section": chunk.section,
                    "chunk_index": chunk.chunk_index,
            }.items()
            if value is not None
            }
            for chunk in chunks 
        ],
    )     