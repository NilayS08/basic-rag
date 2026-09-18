import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.ingestion import chunk_text
from app.vectorstore import VectorStore

TEST_DIM = 16


def fake_embed(text: str) -> list[float]:
    """Deterministic pseudo-embedding: hash-seeded random vector."""
    rng = np.random.default_rng(abs(hash(text)) % (2**32))
    return rng.normal(size=TEST_DIM).tolist()


def test_chunking():
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)
    print(f"[chunking] {len(chunks)} chunks produced from {len(text)} chars — OK")


def test_vectorstore_roundtrip():
    index_dir = "data/test_index"
    shutil.rmtree(index_dir, ignore_errors=True)

    store = VectorStore(index_dir=index_dir, dim=TEST_DIM)

    docs = [
        "The company's refund policy allows returns within 30 days.",
        "Employees get 20 days of paid leave per year.",
        "The office is located in downtown Bangalore.",
    ]
    vectors = [fake_embed(d) for d in docs]
    metas = [{"source_file": "policy.txt", "chunk_id": i, "text": d} for i, d in enumerate(docs)]
    store.add(vectors, metas)

    assert store.count() == 3

    # Query close to doc 0's own embedding (same text -> same fake vector)
    query_vec = fake_embed(docs[0])
    results = store.search(query_vec, top_k=2)
    assert len(results) == 2
    assert results[0]["text"] == docs[0]
    assert results[0]["score"] > 0.99  # near-identical vector should score ~1.0

    print(f"[vectorstore] top match: '{results[0]['text'][:40]}...' score={results[0]['score']:.3f} — OK")

    # Reload from disk to confirm persistence works
    store2 = VectorStore(index_dir=index_dir, dim=TEST_DIM)
    assert store2.count() == 3
    print("[vectorstore] persistence across reload — OK")

    shutil.rmtree(index_dir, ignore_errors=True)


def test_threshold_guardrail_logic():
    """Mirrors the check in rag.answer_question without calling the real LLM."""
    from app.config import SIMILARITY_THRESHOLD

    index_dir = "data/test_index2"
    shutil.rmtree(index_dir, ignore_errors=True)
    store = VectorStore(index_dir=index_dir, dim=TEST_DIM)

    docs = ["The refund window is 30 days."]
    store.add([fake_embed(d) for d in docs], [{"source_file": "x.txt", "chunk_id": 0, "text": docs[0]}])

    # An unrelated query -> random vector, should score low and trigger "not found"
    unrelated_vec = np.random.default_rng(999).normal(size=TEST_DIM).tolist()
    results = store.search(unrelated_vec, top_k=1)
    triggers_not_found = (not results) or (results[0]["score"] < SIMILARITY_THRESHOLD)

    print(f"[guardrail] unrelated query score={results[0]['score']:.3f}, "
          f"threshold={SIMILARITY_THRESHOLD}, would_refuse={triggers_not_found}")

    shutil.rmtree(index_dir, ignore_errors=True)


if __name__ == "__main__":
    test_chunking()
    test_vectorstore_roundtrip()
    test_threshold_guardrail_logic()
    print("\nAll sanity checks passed.")
