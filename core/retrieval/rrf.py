"""Reciprocal rank fusion (Cormack et al. 2009), k = 60."""
from __future__ import annotations


def rrf(rankings: list[list[int]], k: int = 60, top: int | None = None) -> list[tuple[int, float]]:
    """rankings: lists of item ids, best first. Returns (id, fused score) best first."""
    score: dict[int, float] = {}
    for r in rankings:
        for pos, i in enumerate(r[:top] if top else r, start=1):
            score[i] = score.get(i, 0.0) + 1.0 / (k + pos)
    return sorted(score.items(), key=lambda x: (-x[1], x[0]))
