from __future__ import annotations

from typing import Sequence

import numpy as np
import ollama


EMBEDDING_MODEL = "nomic-embed-text"


def embed_texts(
    texts: Sequence[str],
    model: str = EMBEDDING_MODEL,
) -> np.ndarray:
    """
    Generate embeddings for a sequence of texts using Ollama.

    Returns:
        np.ndarray of shape (n_texts, embedding_dimension)
    """

    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    response = ollama.embed(
        model=model,
        input=list(texts),
    )

    embeddings = np.asarray(
        response["embeddings"],
        dtype=np.float32,
    )

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected 2D embeddings, got shape {embeddings.shape}"
        )

    if embeddings.shape[0] != len(texts):
        raise ValueError(
            "Number of embeddings does not match number of input texts"
        )

    return embeddings