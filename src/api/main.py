from pathlib import Path
from tempfile import NamedTemporaryFile
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import RAGServiceError
from src.api.schemas import AskRequest, AskResponse
from src.embeddings.indexer import index_document
from src.ingestion.pdf_loader import load_pdf
from src.rag.service import generate_answer

logger = logging.getLogger(__name__)




app = FastAPI()

# CORS — allow the Next.js frontend (localhost:3000) to call this API.
# Only CORSMiddleware is added; no RAG or API logic is changed.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["POST"],
    allow_headers=["Content-Type"],
)


@app.post("/ask", response_model=AskResponse)


def ask(request: AskRequest):
    try:

        result = generate_answer(
        query=request.question,
        k=5,
        document_id=request.document_id,
    )

    except RAGServiceError:
        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable."
        )


    sources = [
        {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "text": chunk["text"]
        }
        for chunk in result["chunks"]
    ]

    return {
        "status": "answered",
        "answer": result["answer"],
        "model_used": result["model_used"],
        "sources": sources
    }


#DOC UPLOAD Endpoint


@app.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        document = load_pdf(temp_path)

        if not document.sections:
            raise HTTPException(
                status_code=422,
                detail="No readable text found in the PDF.",
            )

        chunk_count = index_document(document)

        return {
            "status": "uploaded_and_indexed",
            "document_id": document.document_id,
            "filename": file.filename,
            "pages": len(document.sections),
            "chunks_indexed": chunk_count,
        }

    except HTTPException:
        raise
    
    except Exception as exc:
        logger.exception("PDF upload failed")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process and index the PDF."
        ) from exc
    
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


    