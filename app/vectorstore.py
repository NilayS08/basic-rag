import json
from pathlib import Path
from typing import Dict, List

import faiss
import numpy as np


class VectorStore:
    def __init__(self, index_dir: str, dim: int):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.index_dir / "index.faiss"
        self.meta_path = self.index_dir / "meta.json"
        self.dim = dim
        self.metadata: List[Dict] = []
        self.index = None
        self._load()

    def _load(self):
        if self.index_path.exists() and self.meta_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.metadata = json.loads(self.meta_path.read_text())
        else:
            self.index = faiss.IndexFlatIP(self.dim)
            self.metadata = []

    def _save(self):
        faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(json.dumps(self.metadata))

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    def add(self, vectors: List[List[float]], metadatas: List[Dict]):
        arr = np.array(vectors, dtype="float32")
        arr = self._normalize(arr)
        self.index.add(arr)
        self.metadata.extend(metadatas)
        self._save()

    def search(self, query_vector: List[float], top_k: int) -> List[Dict]:
        if self.index.ntotal == 0:
            return []

        arr = np.array([query_vector], dtype="float32")
        arr = self._normalize(arr)
        scores, indices = self.index.search(arr, min(top_k, self.index.ntotal))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({**self.metadata[idx], "score": float(score)})
        return results

    def count(self) -> int:
        return self.index.ntotal if self.index is not None else 0
