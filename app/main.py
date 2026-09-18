import logging
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_DIM, INDEX_DIR, UPLOAD_DIR
from app.embeddings import EmbeddingError, embed_texts
from app.ingestion import chunk_text, extract_text
from app.rag import GenerationError, answer_question
from app.schemas import (
    HealthResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
    UploadResponse,
)
from app.vectorstore import VectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Document Q&A RAG System",
    description="Upload PDF/TXT documents and ask questions grounded strictly in their content.",
)

Path(UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
store = VectorStore(index_dir=INDEX_DIR, dim=EMBEDDING_DIM)

ALLOWED_EXTENSIONS = {".pdf", ".txt"}


@app.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Only PDF and TXT are accepted.",
        )

    save_path = Path(UPLOAD_DIR) / file.filename
    content = await file.read()
    save_path.write_bytes(content)

    try:
        text = extract_text(save_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read file: {e}")

    if not text.strip():
        raise HTTPException(status_code=400, detail="No extractable text found in file.")

    chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
    if not chunks:
        raise HTTPException(status_code=400, detail="Document produced no usable chunks.")

    try:
        vectors = embed_texts(chunks, task_type="retrieval_document")
    except EmbeddingError as e:
        raise HTTPException(status_code=502, detail=f"Embedding service failed: {e}")

    metadatas = [
        {"source_file": file.filename, "chunk_id": i, "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
    store.add(vectors, metadatas)

    return UploadResponse(
        filename=file.filename,
        chunks_added=len(chunks),
        total_chunks_in_store=store.count(),
    )


@app.post("/query", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    if store.count() == 0:
        raise HTTPException(status_code=400, detail="No documents uploaded yet.")

    try:
        result = answer_question(store, request.question, request.top_k)
    except GenerationError as e:
        raise HTTPException(status_code=502, detail=str(e))

    return QueryResponse(
        answer=result["answer"],
        grounded=result["grounded"],
        sources=[SourceChunk(**s) for s in result["sources"]],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", documents_indexed=store.count())
