
from pathlib import Path

import chromadb

from src.embeddings.embedder import embed_texts
from src.embeddings.chroma_storage import store_chunks
from src.ingestion.models import Document
from src.preprocessing.chunk_documents import chunk_document


CHROMA_PATH = Path("data/chroma")
COLLECTION_NAME = "documents"


def index_document(document: Document) -> int:
    """
    Chunk, embed, and store an uploaded document.

    Returns the number of chunks indexed.
    """
    chunks = chunk_document(document)

    if not chunks:
        raise ValueError(
            "No readable text was found in the uploaded document."
        )

    embeddings = embed_texts(
        [chunk.text for chunk in chunks]
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_collection(
        name=COLLECTION_NAME
    )

    store_chunks(
        collection=collection,
        chunks=chunks,
        embeddings=embeddings,
    )

    return len(chunks)