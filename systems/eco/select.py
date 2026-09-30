"""Budgeted weighted facility location (brief §5, Selection).

F(S) = sum_i r[i] * max_{m in S} sim[i, m] * fid[m]      subject to  sum_{m in S} cost[m] <= budget

sim must be non-negative, so F is monotone submodular. Solver: lazy cost-benefit greedy (CELF), then return
the better of that solution and the best single feasible item (standard 1/2(1-1/e) guarantee construction).
"""
from __future__ import annotations

import heapq

import numpy as np


def objective(S: list[int], r: np.ndarray, sim: np.ndarray, fid: np.ndarray) -> float:
    if not S:
        return 0.0
    cover = (sim[:, S] * fid[S]).max(axis=1)
    return float(r @ cover)


def select(r: np.ndarray, sim: np.ndarray, fid: np.ndarray, cost: np.ndarray, budget: float,
           group: np.ndarray | None = None) -> tuple[list[int], dict]:
    """group[m]: the unit that candidate m renders. At most one rendering per group is kept; choosing a
    higher-fid rendering of an already selected group replaces it and is charged only the extra cost.
    Default: every candidate is its own group."""
    n = len(r)
    assert sim.shape == (n, n) and (sim >= 0).all(), "sim must be n x n and non-negative"
    group = np.arange(n) if group is None else np.asarray(group)
    members: dict[int, list[int]] = {}
    for m in range(n):
        members.setdefault(int(group[m]), []).append(m)
    cover = np.zeros(n)
    current: dict[int, int] = {}  # group -> selected rendering
    spent = 0.0

    def gain(m: int) -> float:
        return float(r @ np.maximum(sim[:, m] * fid[m] - cover, 0.0))

    def extra(m: int) -> float | None:
        """Cost of adding m now, or None if m cannot improve its group."""
        cur = current.get(int(group[m]))
        if cur is None:
            return float(cost[m])
        if m == cur or fid[m] <= fid[cur]:
            return None
        return max(float(cost[m] - cost[cur]), 1e-6)

    heap = [(-gain(m) / cost[m], m, 0) for m in range(n) if cost[m] <= budget]
    heapq.heapify(heap)
    it = 0
    while heap:
        neg, m, stamp = heapq.heappop(heap)
        c = extra(m)
        if c is None or spent + c > budget:
            continue
        if stamp != it:  # stale bound: recompute and push back
            heapq.heappush(heap, (-gain(m) / c, m, it))
            continue
        if -neg <= 0:
            break
        spent += c
        current[int(group[m])] = m
        cover = np.maximum(cover, sim[:, m] * fid[m])
        it += 1
        for o in members[int(group[m])]:  # upgrade costs dropped: their old bounds are no longer upper bounds
            co = extra(o)
            if co is not None and spent + co <= budget:
                heapq.heappush(heap, (-gain(o) / co, o, it))

    S = sorted(current.values(), key=lambda m: -r[m])
    spent = float(sum(cost[m] for m in S))
    greedy_val = objective(S, r, sim, fid)
    feasible = [m for m in range(n) if cost[m] <= budget]
    best_single = max(feasible, key=lambda m: objective([m], r, sim, fid), default=None)
    single_val = objective([best_single], r, sim, fid) if best_single is not None else 0.0
    if single_val > greedy_val:
        S, spent = [best_single], float(cost[best_single])
    return S, {"objective": max(greedy_val, single_val), "greedy_objective": greedy_val,
               "used_best_single": single_val > greedy_val, "cost": float(spent)}
