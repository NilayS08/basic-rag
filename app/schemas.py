from typing import List, Optional
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    filename: str
    chunks_added: int
    total_chunks_in_store: int


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class SourceChunk(BaseModel):
    source_file: str
    chunk_id: int
    text: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    grounded: bool
    sources: List[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
