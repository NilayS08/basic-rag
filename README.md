# Document Q&A RAG System

Upload PDF/TXT documents, ask questions, get answers grounded strictly in
the uploaded content — with the source chunks returned, and an explicit
refusal when the answer isn't actually in the documents.

## Architecture

```
Upload:  PDF/TXT --> extract_text --> chunk_text --> embed_texts (local all-MiniLM-L6-v2) --> VectorStore (FAISS, local)
Query:   question --> embed_query (local all-MiniLM-L6-v2) --> VectorStore.search (top-k, cosine)
       --> similarity threshold check --> prompt LLM with retrieved chunks --> answer
```


| Component              | File                 | Choice                                                                  |
| ---------------------- | -------------------- | ----------------------------------------------------------------------- |
| API layer              | `app/main.py`        | FastAPI, Pydantic request/response validation                           |
| Document loading       | `app/ingestion.py`   | `pypdf` for PDF, native read for TXT                                    |
| Chunking               | `app/ingestion.py`   | Fixed-size character chunks with overlap                                |
| Embeddings             | `app/embeddings.py`  | Local `all-MiniLM-L6-v2` (no API key) by default; Gemini `text-embedding-004` via `EMBEDDING_MODEL=models/...` with retry + backoff              |
| Vector store           | `app/vectorstore.py` | Local FAISS `IndexFlatIP` (cosine via normalization), persisted to disk |
| Generation + grounding | `app/rag.py`         | Gemini `gemini-1.5-flash`, two-layer refusal guardrail                  |




## Setup

```bash
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and add your GEMINI_API_KEY (get one at https://aistudio.google.com/apikey)

uvicorn app.main:app --reload --port 8000
```

## Usage

**Upload a document**

```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@sample.pdf"
```

**Ask a question**

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Response:

```json
{
  "answer": "...",
  "grounded": true,
  "sources": [
    {"source_file": "sample.pdf", "chunk_id": 3, "text": "...", "score": 0.78}
  ]
}
```

If the answer isn't in the documents, `grounded` is `false`, `sources` is
empty, and `answer` explicitly says so instead of guessing.

**Health check**

```bash
curl http://localhost:8000/health
```



## Configuration

All tunable parameters live in `app/config.py` and can be overridden via
environment variables (see `.env.example`): `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`TOP_K`, `SIMILARITY_THRESHOLD`, `EMBEDDING_MODEL`, `GENERATION_MODEL`.
Embeddings run fully offline by default (a local sentence-transformers model,
downloaded once on first use); set `EMBEDDING_MODEL=models/text-embedding-004`
to switch to the Gemini embedding API. Note `SIMILARITY_THRESHOLD` is
embedding-model-dependent: the default `0.35` is calibrated for
`all-MiniLM-L6-v2` score distributions, not Gemini's.

## Testing without an API key

`tests/test_pipeline.py` verifies chunking, FAISS storage/retrieval, and
the threshold-guardrail logic using deterministic fake embeddings — no
network or API key required:

```bash
python3 tests/test_pipeline.py
```



## What works

- PDF and TXT upload, chunking, embedding, and local persistent storage
- Retrieval + grounded generation with citations back to source chunks
- Explicit "not found" response when a question falls outside the
uploaded documents (two-layer guardrail — see explanation doc)
- Input validation (file type, empty files, empty questions) with
appropriate HTTP error codes
- Error handling around embedding and generation (embeddings run locally
by default; retry + backoff applies to the Gemini embedding API, clean
error surfacing on generation)



## What doesn't / known limitations

- Single global document store — no per-user or per-session isolation
(out of scope per task spec)
- No support for scanned/image-only PDFs (no OCR step)
- Character-based chunking can split mid-sentence; a token- or
sentence-aware chunker would be more precise but adds complexity
- No de-duplication if the same file is uploaded twice (it will be
indexed again as a separate set of chunks)
- See the explanation document for what was deliberately left unfinished
and why.

