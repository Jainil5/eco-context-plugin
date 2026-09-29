"""Dense retrieval: cosine similarity over cached embeddings (EmbeddingClient)."""
from __future__ import annotations

import numpy as np


def normalize(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(n == 0, 1, n)


class DenseIndex:
    def __init__(self, texts: list[str], embedder, stage: str = "construction", qid: str | None = None):
        self.embedder = embedder
        self.vecs = normalize(np.asarray(embedder.embed(texts, stage=stage, qid=qid), dtype=np.float32))

    def query_vec(self, query: str, qid: str | None = None) -> np.ndarray:
        return normalize(np.asarray(self.embedder.embed([query], stage="query_time", qid=qid), dtype=np.float32))[0]

    def rank(self, query: str, qid: str | None = None) -> list[tuple[int, float]]:
        s = self.vecs @ self.query_vec(query, qid)
        return sorted(enumerate(s.tolist()), key=lambda x: (-x[1], x[0]))
