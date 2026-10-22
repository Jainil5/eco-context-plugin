"""Maximal marginal relevance (Carbonell & Goldstein 1998) over an explicit relevance vector and embedding matrix.

Used by the turn-level and span-level MMR baselines: the only difference between them is what the candidates are.
"""
from __future__ import annotations

import numpy as np


def mmr_order(rel: np.ndarray, V: np.ndarray, lam: float) -> list[int]:
    """Greedy relevance-minus-redundancy order over every candidate, best first. Tie-break: lowest index."""
    n = len(rel)
    order: list[int] = []
    left = set(range(n))
    max_sim = np.zeros(n)
    while left:
        i = max(left, key=lambda j: (lam * rel[j] - (1 - lam) * max_sim[j], -j))
        order.append(i)
        left.discard(i)
        max_sim = np.maximum(max_sim, V @ V[i])
    return order
