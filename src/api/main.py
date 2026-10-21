from pathlib import Path
from tempfile import NamedTemporaryFile
import logging
from uuid import uuid4 

from fastapi import FastAPI, HTTPException, UploadFile, File , BackgroundTasks 
from fastapi.middleware.cors import CORSMiddleware

from src.api.exceptions import RAGServiceError
from src.api.schemas import AskRequest, AskResponse
from src.embeddings.indexer import index_document
from src.ingestion.pdf_loader import load_pdf
from src.ingestion.docx_loader import load_docx 
from src.rag.service import generate_answer
from evaluation.evaluate_uploaded_docs import evaluate_uploaded_document 

logger = logging.getLogger(__name__)

# Temporary in-memory conversation storage
conversation_store: dict[str, list[dict[str, str]]] = {}

MAX_HISTORY_MESSAGES = 10

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
    # Reuse the conversation or create a new one.
    conversation_id = request.conversation_id or str(uuid4())

    # Load previous messages from server-side memory.
    history = conversation_store.get(conversation_id, [])
    print("Conversation history: ", history)

    try:
        result = generate_answer(
            query=request.question,
            k=5,
            document_id=request.document_id,
            conversation_history=history,
        )
    except RAGServiceError as exc:
        logger.exception("RAG Service Error")
        raise HTTPException(
            status_code=503,
            detail="RAG service is temporarily unavailable.",
        ) from exc

    sources = [
        {
            "doc_id": chunk["doc_id"],
            "chunk_id": chunk["chunk_id"],
            "source": chunk["source"],
            "filename": chunk.get("filename"),
            "page": chunk.get("page"),
            "section": chunk.get("section"),
            "citation": (
                f'{chunk["filename"]} - Page {chunk["page"]}'
                if chunk.get("filename")
                and chunk.get("page") is not None
                else (
                    f'{chunk["filename"]} - {chunk["section"]}'
                    if chunk.get("filename") and chunk.get("section")
                    else chunk.get("filename") or chunk["source"]
                )
            ),
            "text": chunk["text"],
        }
        for chunk in result["chunks"]
    ]

    # Save the new exchange in the conversation history.
    updated_history = history + [
        {"role": "user", "content": request.question},
        {"role": "assistant", "content": result["answer"]},
    ]

    conversation_store[conversation_id] = updated_history[
        -MAX_HISTORY_MESSAGES:
    ]
    print("saving conversation ID: ", conversation_id)
    print("stored history: " , conversation_store.get(conversation_id))
    return {
        "status": "answered",
        "answer": result["answer"],
        "model_used": result["model_used"],
        "sources": sources,
        "conversation_id": conversation_id,
    }

#DOC UPLOAD Endpoint


@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(...)):
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()

    loaders = {
        ".pdf": load_pdf,
        ".docx": load_docx,
    }

    if suffix not in loaders:
        raise HTTPException(
            status_code=400,
            detail="Only PDF and DOCX files are supported.",
        )

    temp_path = None

    try:
        with NamedTemporaryFile(
            suffix=suffix,
            delete=False,
        ) as temp_file:
            temp_file.write(await file.read())
            temp_path = Path(temp_file.name)

        document = loaders[suffix](
            temp_path,
            original_filename=filename,
        )

        if not document.sections:
            raise HTTPException(
                status_code=422,
                detail="No readable text found in the document.",
            )

        chunk_count = index_document(document)

        background_tasks.add_task(
            evaluate_uploaded_document , 
            document.document_id,
        )

        return {
            "status": "uploaded_and_indexed",
            "document_id": document.document_id,
            "filename": filename,
            "pages": (
                len(document.sections)
                if suffix == ".pdf"
                else None
            ),
            "chunks_indexed": chunk_count,
        }

    except HTTPException:
        raise

    except Exception as exc:
        logger.exception("Document upload failed")
        raise HTTPException(
            status_code=500,
            detail="Failed to process and index the document.",
        ) from exc

    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)



    