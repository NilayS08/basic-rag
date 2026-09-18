import logging
import time
from typing import List

from app.config import EMBEDDING_MODEL, GEMINI_API_KEY

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _LOCAL_SUPPORTED = True
except ImportError:  # pragma: no cover
    _LOCAL_SUPPORTED = False
    SentenceTransformer = None

import google.generativeai as genai

_genai_configured = False
_local_model = None


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""


def _is_gemini_model(model: str) -> bool:
    return model.startswith("models/") or model.startswith("gemini-")


def _embed_local(texts: List[str]) -> List[List[float]]:
    global _local_model
    if not _LOCAL_SUPPORTED:
        raise EmbeddingError(
            "sentence-transformers is not installed. Run: pip install sentence-transformers"
        )
    if _local_model is None:
        logger.info("Loading local embedding model %s (first use downloads it)...", EMBEDDING_MODEL)
        _local_model = SentenceTransformer(EMBEDDING_MODEL)
    try:
        vectors = _local_model.encode(texts, normalize_embeddings=True)
        return [list(map(float, vec)) for vec in vectors]
    except Exception as e:
        raise EmbeddingError(f"Local embedding failed: {e}")


def _embed_gemini(
    texts: List[str],
    task_type: str,
    max_retries: int = 3,
) -> List[List[float]]:
    global _genai_configured
    if not GEMINI_API_KEY:
        raise EmbeddingError("GEMINI_API_KEY is not set")
    if not _genai_configured:
        genai.configure(api_key=GEMINI_API_KEY)
        _genai_configured = True

    embeddings = []
    for text in texts:
        last_err = None
        for attempt in range(1, max_retries + 1):
            try:
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=text,
                    task_type=task_type,
                )
                embeddings.append(result["embedding"])
                last_err = None
                break
            except Exception as e:  # network errors, rate limits, etc.
                last_err = e
                logger.warning("Embedding attempt %d failed: %s", attempt, e)
                time.sleep(1.5 * attempt)  # simple linear backoff

        if last_err is not None:
            raise EmbeddingError(
                f"Failed to embed text after {max_retries} attempts: {last_err}"
            )

    return embeddings


def embed_texts(
    texts: List[str],
    task_type: str = "retrieval_document",
    max_retries: int = 3,
) -> List[List[float]]:
    if not texts:
        return []

    model = EMBEDDING_MODEL
    logger.info(
        "Embedding %d texts with %s (%s)",
        len(texts),
        model,
        "Gemini API" if _is_gemini_model(model) else "local sentence-transformers",
    )

    if _is_gemini_model(model):
        return _embed_gemini(texts, task_type=task_type, max_retries=max_retries)
    return _embed_local(texts)


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]