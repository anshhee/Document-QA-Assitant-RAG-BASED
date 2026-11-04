# Biomedical Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) platform for querying biomedical literature, clinical-trial records, and uploaded PDF/DOCX documents using natural-language questions. Answers are grounded in retrieved evidence and returned with source citations.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Current Implementation Status](#2-current-implementation-status)
3. [Technology Stack](#3-technology-stack)
4. [System Architecture](#4-system-architecture)
5. [End-to-End RAG Pipeline](#5-end-to-end-rag-pipeline)
6. [Chunk Configuration](#6-chunk-configuration)
7. [Conversation Support](#7-conversation-support)
8. [FastAPI Endpoints](#8-fastapi-endpoints)
9. [Streamlit Frontend](#9-streamlit-frontend)
10. [RAGAS Evaluation](#10-ragas-evaluation)
11. [Project Folder Structure](#11-project-folder-structure)
12. [Installation and Setup](#12-installation-and-setup)
13. [Ollama Model Setup](#13-ollama-model-setup)
14. [Running the Project](#14-running-the-project)
15. [Environment Variables](#15-environment-variables)
16. [Known Limitations](#16-known-limitations)
17. [Roadmap](#17-roadmap)
18. [Future Improvements](#18-future-improvements)
19. [Disclaimer](#19-disclaimer)

---

## 1. Project Overview

This project is a practice implementation of a full RAG (Retrieval-Augmented Generation) pipeline applied to the biomedical domain. Users upload PDF or DOCX documents through a Streamlit frontend, which sends them to a FastAPI backend for parsing, chunking, embedding, and ChromaDB indexing. Users then ask natural-language questions; the system retrieves the most semantically relevant document chunks and passes them as evidence to a language model, which generates a grounded answer with source citations.

**Key design goals:**

- Ground every answer strictly in retrieved evidence — the model is instructed not to guess or add unsupported facts.
- Preserve scientific precision: distinguish drugs, isotopes, clinical trials, study populations, and dosing details.
- Support multi-turn conversations with server-side history.
- Automatically evaluate each uploaded document with RAGAS faithfulness and answer-relevancy metrics in the background.

---

## 2. Current Implementation Status

| Feature | Status |
|---|---|
| PDF and DOCX upload and parsing | ✅ Active |
| OCR fallback for image-only PDF pages (Tesseract) | ✅ Active |
| Text cleaning and chunking | ✅ Active |
| 1 000-character chunks with 200-character overlap | ✅ Active |
| `nomic-embed-text` embeddings via Ollama | ✅ Active |
| ChromaDB vector storage | ✅ Active |
| Semantic similarity retrieval (top-5 chunks) | ✅ Active |
| Document-scoped retrieval by `document_id` | ✅ Active |
| Qwen 2.5:3b answer generation (uploaded documents) | ✅ Active |
| Gemini answer generation with Qwen fallback (biomedical corpus) | ✅ Active |
| Follow-up query contextualization | ✅ Active |
| Conversation IDs and server-side history | ✅ Active |
| Source citations (filename, page, section) | ✅ Active |
| Streamlit frontend | ✅ Active |
| FastAPI backend | ✅ Active |
| Background RAGAS evaluation on upload | ✅ Active |
| Faithfulness metric | ✅ Active |
| Answer-relevancy metric | ✅ Active |
| ChromaDB threading lock | ✅ Active |
| Lexical reranking function | ⚠️ Exists but NOT connected to production retrieval |
| Hybrid search | 🔲 Not implemented |
| Dockerization / deployment | 🔲 Not implemented |

> [!NOTE]
> The current production retrieval strategy is **semantic search only**. A `rerank_chunks()` function exists in `src/rag/service.py` and uses basic term-overlap scoring, but it is not wired into the retrieval flow. Hybrid retrieval and cross-encoder reranking are future improvements.

---

## 3. Technology Stack

| Component | Technology |
|---|---|
| Language | Python 3.12+ |
| API framework | FastAPI |
| Frontend | Streamlit |
| Vector store | ChromaDB (persistent, local) |
| Local model runtime | Ollama |
| Embedding model | `nomic-embed-text` (via Ollama) |
| Primary generation model | Qwen 2.5:3b (via Ollama) |
| Cloud generation model | Gemini (via Google GenAI SDK, optional) |
| PDF parsing | PyMuPDF (`fitz`) |
| OCR fallback | Tesseract + Pillow |
| DOCX parsing | `python-docx` |
| RAG evaluation | RAGAS |
| LLM bridge for RAGAS | LiteLLM (via `embedding_factory` with `ollama/` prefix) |
| Pydantic models | Pydantic v2 |
| Environment variables | `python-dotenv` |

---

## 4. System Architecture

```text
┌─────────────────────────────────────────────────────────────────┐
│                         User (Browser)                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │  HTTP (localhost:8501)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Streamlit Frontend                          │
│  • File uploader (PDF / DOCX)                                   │
│  • Conversation session management                               │
│  • Chat input / message history                                  │
│  • Source citations expander                                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │  HTTP (localhost:8000)
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                       FastAPI Backend                            │
│                                                                 │
│  POST /documents/upload          POST /ask                      │
│  ┌─────────────────────┐         ┌──────────────────────────┐  │
│  │ 1. Parse PDF / DOCX │         │ 1. Load conversation     │  │
│  │ 2. Clean text       │         │    history               │  │
│  │ 3. Chunk text       │         │ 2. Contextualize query   │  │
│  │ 4. Embed chunks     │         │ 3. Semantic retrieval    │  │
│  │ 5. Store in Chroma  │         │ 4. Build prompt          │  │
│  │ 6. Trigger RAGAS    │         │ 5. Generate answer       │  │
│  │    evaluation       │         │ 6. Save history          │  │
│  └─────────────────────┘         │ 7. Return answer +       │  │
│                                  │    sources               │  │
│                                  └──────────────────────────┘  │
└──────┬──────────────────────────────────┬───────────────────────┘
       │                                  │
       ▼                                  ▼
┌──────────────┐                 ┌─────────────────────┐
│   ChromaDB   │                 │       Ollama         │
│  (data/chroma│                 │  nomic-embed-text    │
│  persistent) │                 │  qwen2.5:3b          │
└──────────────┘                 └─────────────────────┘
                                          │
                                          ▼ (optional fallback)
                                 ┌─────────────────────┐
                                 │   Google Gemini API  │
                                 │  (if GEMINI_API_KEY  │
                                 │   is configured)     │
                                 └─────────────────────┘
```

---

## 5. End-to-End RAG Pipeline

### 5.1 Document Ingestion Pipeline

```
Uploaded file (PDF or DOCX)
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  Parsing                                            │
  │  • PDF:  PyMuPDF extracts embedded text per page   │
  │          Tesseract OCR used if page has no text     │
  │  • DOCX: python-docx extracts paragraphs/headings  │
  │  Result: Document object with sections + metadata  │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  Cleaning                                           │
  │  • Empty sections are dropped                       │
  │  • Whitespace and formatting noise removed          │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  Chunking  (src/preprocessing/chunk_documents.py)  │
  │  • Chunk size: 1 000 characters                    │
  │  • Overlap:    200 characters                      │
  │  • Boundaries: paragraph > sentence > hard cut     │
  │  • Each chunk carries: chunk_id, document_id,      │
  │    filename, page, section, chunk_index            │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  Embedding  (src/embeddings/embedder.py)            │
  │  • Model: nomic-embed-text via Ollama               │
  │  • Each chunk text is embedded independently        │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  ┌─────────────────────────────────────────────────────┐
  │  ChromaDB Indexing  (src/embeddings/chroma_storage) │
  │  • Collection: "documents" (persistent)             │
  │  • Stored: embeddings + text + metadata             │
  │  • Unique IDs: {document_id}_{chunk_index}          │
  └─────────────────────────────────────────────────────┘
        │
        ▼
  RAGAS evaluation triggered in background
```

### 5.2 Retrieval and Answer Generation Pipeline

```
User question
      │
      ▼
  Contextualize query
  (Qwen rewrites follow-up questions into standalone queries)
      │
      ▼
  Embed query  →  nomic-embed-text
      │
      ▼
  ChromaDB semantic search
  • Returns top-5 chunks by cosine similarity
  • Filtered by document_id when a document is active
      │
      ▼
  Build prompt
  • Conversation history injected
  • Retrieved evidence injected as [Source N] blocks
  • Detailed instruction set for scientific accuracy
      │
      ▼
  Answer generation
  • Uploaded documents  →  Qwen 2.5:3b (local, via Ollama)
  • Biomedical corpus   →  Gemini (with Qwen fallback)
      │
      ▼
  Response returned with:
  • answer
  • sources  (doc_id, chunk_id, filename, page, section, citation, text)
  • model_used
  • conversation_id
```

### 5.3 Source Citations

Each answer includes a `sources` list. Citations are formatted as:

| Available metadata | Citation format |
|---|---|
| Filename + page number | `report.pdf - Page 3` |
| Filename + section heading | `report.docx - Introduction` |
| Filename only | `report.pdf` |
| Fallback | raw `source` field |

---

## 6. Chunk Configuration

| Parameter | Value |
|---|---|
| Chunk size | 1 000 characters |
| Overlap | 200 characters |
| Boundary preference | Paragraph → Sentence → Hard cut |
| Boundary search window | ±200 characters around target |

Chunking is page-aware: each page or DOCX section is chunked independently, so chunks never span page boundaries. Each chunk records its `page` number and `section` heading for citation.

---

## 7. Conversation Support

The backend maintains server-side conversation history in an in-memory dictionary (`conversation_store`).

| Feature | Detail |
|---|---|
| Conversation IDs | UUID4, auto-generated on first question if not supplied |
| History storage | In-memory per process (resets on server restart) |
| Max history length | 10 messages (5 exchanges) |
| Follow-up rewriting | Qwen rewrites ambiguous follow-ups into standalone retrieval queries using the last 6 messages of history |
| History in prompt | Full conversation history is injected into the generation prompt for context |

**Request fields for `/ask`:**

```json
{
  "question": "What is the mechanism of action?",
  "document_id": "uuid-of-uploaded-document",   // optional — scopes retrieval
  "conversation_id": "uuid-of-conversation"     // optional — omit to start new
}
```

---

## 8. FastAPI Endpoints

### `POST /documents/upload`

Upload and index a PDF or DOCX document.

**Request:** `multipart/form-data` with field `file`.

**Supported file types:** `.pdf`, `.docx`

**Response (200):**

```json
{
  "status": "uploaded_and_indexed",
  "document_id": "3f7a1c2e-...",
  "filename": "trial_report.pdf",
  "pages": 12,
  "chunks_indexed": 47
}
```

**Error responses:**

| Code | Condition |
|---|---|
| 400 | File type not `.pdf` or `.docx` |
| 422 | No readable text found in the document |
| 500 | Internal parsing or indexing failure |

**Side effect:** A RAGAS evaluation job is queued as a FastAPI `BackgroundTask` immediately after indexing.

---

### `POST /ask`

Ask a question against the indexed corpus or a specific uploaded document.

**Request body (`AskRequest`):**

```json
{
  "question": "string (1–2000 characters, required)",
  "document_id": "string | null",
  "conversation_id": "string | null"
}
```

**Response body (`AskResponse`):**

```json
{
  "status": "answered",
  "answer": "string",
  "model_used": "qwen2.5:3b | gemini | none",
  "conversation_id": "string",
  "sources": [
    {
      "doc_id": "string",
      "chunk_id": "string",
      "source": "string",
      "filename": "string | null",
      "page": "integer | null",
      "section": "string | null",
      "citation": "string | null",
      "text": "string"
    }
  ]
}
```

**Error responses:**

| Code | Condition |
|---|---|
| 503 | RAG service temporarily unavailable (Ollama/Gemini unreachable) |

---

## 9. Streamlit Frontend

The Streamlit app (`streamlit-app/streamlit_app.py`) connects to the FastAPI backend at `http://127.0.0.1:8000` by default (overridable via the `RAG_API_URL` environment variable).

**Workflow:**

1. **Upload** — User selects a PDF or DOCX using the file uploader. The app `POST`s to `/documents/upload` and stores the returned `document_id` in session state. A new conversation session is created automatically.
2. **Session management** — The sidebar lists all conversation sessions. Users can start a new session with "+ New session". Session titles default to the first 40 characters of the opening question.
3. **Ask a question** — The chat input field submits a question via `POST /ask`, including the `document_id` and `conversation_id`. The spinner shows "Retrieving evidence..." during the request.
4. **View answer** — The answer is rendered as Markdown in a chat message bubble.
5. **View sources** — An expandable "References" section appears below each answer showing the citation label and chunk text for each source.

> [!IMPORTANT]
> A document must be uploaded before questions can be asked. The app shows "Upload a PDF or DOCX above to begin asking questions." until a document is indexed.

---

## 10. RAGAS Evaluation

### How it works

After every successful document upload, the FastAPI server schedules `evaluate_uploaded_document(document_id)` as a background task. This runs asynchronously without blocking the upload response.

**Evaluation steps:**

1. All chunks belonging to the uploaded document are loaded from ChromaDB.
2. Qwen 2.5:3b generates **5 evaluation questions** from a sample of up to 8 chunks, spread evenly across the document.
3. For each question, the RAG pipeline (`generate_answer`) is called to produce an answer and retrieve contexts.
4. RAGAS scores each question–answer pair.
5. Results are saved to `evaluation_results/uploaded_documents_ragas_results[_N].json`.

### Metrics

| Metric | Description |
|---|---|
| **Faithfulness** | Measures whether claims in the answer are supported by the retrieved contexts. Score range: 0–1. |
| **Answer Relevancy** | Measures whether the answer addresses the question. Score range: 0–1. |

### RAGAS LLM and embedding configuration

| Component | Value |
|---|---|
| LLM provider | `openai` (OpenAI-compatible Ollama endpoint at `http://localhost:11434/v1`) |
| LLM model | `qwen2.5:3b` |
| Embedding provider | LiteLLM (`ollama/nomic-embed-text` at `http://localhost:11434`) |

### Output format

```json
{
  "document_id": "...",
  "num_questions": 5,
  "summary": {
    "average_faithfulness": 0.84,
    "average_answer_relevancy": 0.91
  },
  "results": [
    {
      "question": "...",
      "answer": "...",
      "faithfulness": 0.9,
      "answer_relevancy": 0.95,
      "document_id": "...",
      "model_used": "qwen2.5:3b",
      "contexts": ["...", "..."],
      "num_contexts": 5,
      "retrieved_chunks": [...]
    }
  ]
}
```

### Background evaluation and ChromaDB locking

Both the main request handler and the background evaluator may access ChromaDB concurrently. A `threading.Lock` (`collection_lock`) in `src/rag/service.py` serializes all ChromaDB reads and writes to prevent data races.

---

## 11. Project Folder Structure

```
Practice rag project/
├── .env                              # Environment variables (GEMINI_API_KEY)
├── .gitignore
├── README.md
├── requirements.txt
│
├── data/
│   ├── chroma/                       # Persistent ChromaDB storage
│   └── processed/
│       └── chunked_documents.json    # Pre-chunked biomedical corpus metadata
│
├── evaluation/
│   └── evaluate_uploaded_docs.py     # RAGAS evaluation pipeline
│
├── evaluation_results/               # RAGAS output JSON files (auto-created)
│   └── uploaded_documents_ragas_results[_N].json
│
├── notebooks/
│   └── rag_pipeline.ipynb            # Exploratory notebook
│
├── src/
│   ├── api/
│   │   ├── exceptions.py             # RAGServiceError
│   │   ├── main.py                   # FastAPI app, /ask, /documents/upload
│   │   └── schemas.py                # Pydantic request/response models
│   │
│   ├── embeddings/
│   │   ├── chroma_storage.py         # ChromaDB write helper
│   │   ├── embedder.py               # Ollama nomic-embed-text wrapper
│   │   └── indexer.py                # Chunk → embed → store pipeline
│   │
│   ├── ingestion/
│   │   ├── clinical_trials.py        # ClinicalTrials.gov data loader
│   │   ├── docx_loader.py            # python-docx DOCX loader
│   │   ├── europe_pmc.py             # Europe PMC loader
│   │   ├── models.py                 # Document, DocumentSection, DocumentChunk
│   │   └── pdf_loader.py             # PyMuPDF + Tesseract OCR PDF loader
│   │
│   ├── preprocessing/
│   │   ├── chunk_documents.py        # Chunking logic (1 000 / 200)
│   │   └── clean_documents.py        # Text cleaning utilities
│   │
│   └── rag/
│       └── service.py                # retrieve_chunks, build_prompt,
│                                     # generate_answer, rerank_chunks (unused)
│
├── streamlit-app/
│   └── streamlit_app.py              # Streamlit frontend
│
└── tests/
    ├── test_api.py
    └── test_rag.py
```

> [!NOTE]
> A `frontend/` directory also exists (Next.js scaffold) but is not the active frontend. The production frontend is Streamlit.

---

## 12. Installation and Setup

### Prerequisites

| Requirement | Notes |
|---|---|
| Python 3.12+ | Tested on 3.12 and 3.14 |
| Ollama | [https://ollama.com](https://ollama.com) |
| Tesseract OCR | Required for OCR fallback on image-only PDF pages |
| Git | For cloning |

### Steps

```bash
# 1. Clone the repository
git clone <repo-url>
cd "Practice rag project"

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows (PowerShell)
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

> [!NOTE]
> `requirements.txt` currently lists no packages (they were installed incrementally). Install these packages manually if starting fresh:
>
> ```
> fastapi uvicorn[standard] streamlit chromadb ollama pymupdf
> python-docx pillow pytesseract python-dotenv pydantic ragas
> litellm openai google-genai requests
> ```

### Tesseract OCR installation

| Platform | Instructions |
|---|---|
| **Windows** | Download installer from [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki). The loader expects Tesseract at `C:\Program Files\Tesseract-OCR\tesseract.exe`. |
| **macOS** | `brew install tesseract` |
| **Linux** | `sudo apt-get install tesseract-ocr` |

---

## 13. Ollama Model Setup

Ollama must be running before starting the backend.

```bash
# Start Ollama (runs on http://localhost:11434 by default)
ollama serve

# Pull the embedding model
ollama pull nomic-embed-text

# Pull the generation model
ollama pull qwen2.5:3b
```

Verify models are available:

```bash
ollama list
```

---

## 14. Running the Project

Open three separate terminal windows, each with the virtual environment activated.

### FastAPI backend

```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```

The API is available at `http://127.0.0.1:8000`.  
Interactive docs: `http://127.0.0.1:8000/docs`

### Streamlit frontend

```bash
streamlit run streamlit-app/streamlit_app.py
```

The UI is available at `http://localhost:8501`.

### RAGAS evaluator (manual / standalone)

The evaluator runs automatically as a background task on each upload. To run it manually for a specific document:

```bash
python evaluation/evaluate_uploaded_docs.py <document_id>
```

Replace `<document_id>` with the UUID returned by `/documents/upload`.

---

## 15. Environment Variables

Create a `.env` file in the project root:

```dotenv
# Optional — enables Gemini as the primary generation model for biomedical queries.
# If not set, Qwen 2.5:3b is used for all generation.
GEMINI_API_KEY=your_key_here
```

The Streamlit app reads `RAG_API_URL` from the environment to override the backend address:

```dotenv
# Default: http://127.0.0.1:8000
RAG_API_URL=http://127.0.0.1:8000
```

---

## 16. Known Limitations

| Limitation | Detail |
|---|---|
| In-memory conversation history | Conversation history is stored in process memory and is lost on server restart. |
| No authentication | All endpoints are public. Any caller on the network can upload documents or query the system. |
| Tesseract path is hardcoded | `pdf_loader.py` sets the Tesseract path to `C:\Program Files\Tesseract-OCR\tesseract.exe`. This must be changed on non-Windows systems. |
| Single ChromaDB collection | All uploaded documents share the `"documents"` collection. There is no per-user or per-session isolation. |
| RAGAS uses local Qwen for evaluation | Evaluation quality depends on the 3B model's ability to follow complex scoring instructions. |
| No document deletion endpoint | Uploaded documents and their chunks persist in ChromaDB indefinitely. |
| Lexical reranker not connected | `rerank_chunks()` exists but is not called during production retrieval. |
| `requirements.txt` is empty | Dependencies must be installed manually. |
| Gemini model name | The code references `gemini-3.5-flash` — verify this matches the available Gemini model names in your API access tier. |

---

## 17. Roadmap

### ✅ Completed

- PDF and DOCX parsing (PyMuPDF + python-docx)
- OCR fallback for image-only PDF pages
- Text cleaning and boundary-aware chunking (1 000 / 200)
- `nomic-embed-text` embeddings via Ollama
- ChromaDB persistent vector store
- Semantic retrieval (top-5 chunks, document-scoped)
- Qwen 2.5:3b answer generation
- Gemini generation with Qwen fallback
- Follow-up query contextualization
- Server-side conversation history (max 10 messages)
- Source citations with page and section metadata
- FastAPI backend with `/ask` and `/documents/upload`
- Streamlit frontend (file upload, chat, citations)
- Background RAGAS evaluation (Faithfulness + Answer Relevancy)
- ChromaDB threading lock for concurrent access

### 🔄 In Progress / Partial

- Lexical reranking (`rerank_chunks()` implemented, not connected)
- Tests (`tests/test_api.py`, `tests/test_rag.py` exist; coverage not measured)

### 🔲 Planned

- Connect or replace lexical reranker with a cross-encoder reranker
- Hybrid retrieval (BM25 + semantic vector search)
- Persistent conversation history (database-backed)
- Document deletion endpoint
- Authentication and per-user document isolation
- Additional RAGAS metrics (context precision, context recall)
- Configurable `k` and chunk parameters via API
- Containerisation (Docker Compose)

---

## 18. Future Improvements

- **Cross-encoder reranking** — Replace or supplement the existing term-overlap reranker with a cross-encoder model (e.g., `ms-marco-MiniLM`) to improve retrieval precision.
- **Hybrid search** — Combine semantic vector search with BM25 keyword search for better recall on exact biomedical terminology (drug names, gene symbols, trial IDs).
- **Persistent conversation store** — Replace the in-memory dictionary with a Redis or SQLite-backed store so conversations survive server restarts.
- **Streaming answers** — Stream token-by-token output from Ollama to reduce perceived latency for long answers.
- **Configurable retrieval parameters** — Expose `k` (number of retrieved chunks) and chunk size/overlap via API request fields.
- **Document management** — Add endpoints to list uploaded documents, delete them, and browse their indexed chunks.
- **Additional RAGAS metrics** — Add context precision, context recall, and context entity recall once a reference dataset is available.
- **Multi-modal support** — Extend the pipeline to index and retrieve from figures and tables extracted from PDFs.
- **Deployment** — Containerise all services with Docker Compose (FastAPI + Streamlit + ChromaDB + Ollama).

---

## 19. Disclaimer

> [!CAUTION]
> **For educational use only.**
>
> This system is a learning project demonstrating RAG techniques on biomedical text. It is **not** a certified medical device, clinical decision-support system, or validated diagnostic tool.
>
> - Do **not** use this system to make medical, clinical, or treatment decisions.
> - Answers are generated by a language model and may contain errors, omissions, or hallucinations even when grounded in retrieved evidence.
> - Always consult a qualified healthcare professional for medical advice.
> - The retrieved evidence is limited to the documents you upload and may not reflect the full body of relevant scientific literature.
>
> The developers make no warranties, express or implied, regarding the accuracy, completeness, or fitness for any purpose of the information generated by this system.
