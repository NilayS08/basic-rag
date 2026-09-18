import logging
from typing import Dict, List, Optional

import google.generativeai as genai

from app.config import (
    GEMINI_API_KEY,
    GENERATION_MODEL,
    SIMILARITY_THRESHOLD,
    TOP_K,
)
from app.embeddings import EmbeddingError, embed_query
from app.vectorstore import VectorStore

logger = logging.getLogger(__name__)

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

NOT_FOUND_MESSAGE = "I can't find an answer to this in the uploaded documents."

PROMPT_TEMPLATE = """You are answering strictly using the CONTEXT below.
If the answer is not fully supported by the context, respond with exactly
this sentence and nothing else: "{not_found}"
Do not use outside knowledge. Do not guess or fill gaps.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""


class GenerationError(Exception):
    """Raised when embedding the query or calling the generation model fails."""


def _build_context(chunks: List[Dict]) -> str:
    parts = []
    for i, c in enumerate(chunks, start=1):
        parts.append(
            f"[{i}] (source: {c['source_file']}, chunk {c['chunk_id']})\n{c['text']}"
        )
    return "\n\n".join(parts)


def answer_question(
    store: VectorStore, question: str, top_k: Optional[int] = None
) -> Dict:
    top_k = top_k or TOP_K

    try:
        query_vec = embed_query(question)
    except EmbeddingError as e:
        raise GenerationError(f"Could not embed the query: {e}")

    retrieved = store.search(query_vec, top_k)

    if not retrieved or retrieved[0]["score"] < SIMILARITY_THRESHOLD:
        return {"answer": NOT_FOUND_MESSAGE, "grounded": False, "sources": []}

    context = _build_context(retrieved)
    prompt = PROMPT_TEMPLATE.format(
        context=context, question=question, not_found=NOT_FOUND_MESSAGE
    )

    try:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(prompt)
        answer_text = (response.text or "").strip()
    except Exception as e:
        logger.error("Generation failed: %s", e)
        raise GenerationError(f"LLM generation failed: {e}")

    grounded = NOT_FOUND_MESSAGE not in answer_text

    return {
        "answer": answer_text,
        "grounded": grounded,
        "sources": retrieved if grounded else [],
    }
