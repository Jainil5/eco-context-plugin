"""Budgeted weighted facility location (brief §5, Selection).

F(S) = sum over demand units i of  r(i) * max over m in S of sim(i, m) * fid(m)      subject to sum cost(m) <= B

sim must be non-negative, so F is monotone submodular. Solver: lazy cost-benefit greedy (CELF), then return
the better of that solution and the best single feasible item (standard 1/2(1-1/e) guarantee construction).

Two demand shapes are allowed on purpose:
  * square (n x n): every candidate is also a demand unit - the original objective, and the selection order is the
    candidate's own relevance r[m], which is byte-identical to the published turn-level runs.
  * rectangular (D x n, D != n): extra demand rows that cannot be selected, used by multi-facet coverage where a
    query facet must be covered but is not something the reader can be shown. There the selection order falls back
    to each candidate's own total coverage.
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
    n = len(cost)                                   # number of selectable candidates
    assert len(r) == sim.shape[0] and sim.shape[1] == n and len(fid) == n, \
        f"demand has {len(r)} rows, sim is {sim.shape}, but there are {n} candidates"
    assert (sim >= 0).all(), "sim must be non-negative"
    square = len(r) == n
    group = np.arange(n) if group is None else np.asarray(group)
    members: dict[int, list[int]] = {}
    for m in range(n):
        members.setdefault(int(group[m]), []).append(m)
    cover = np.zeros(len(r))
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

    # Square demand: a candidate's own relevance orders the set (original behaviour). Otherwise rank by how much
    # each candidate covers on its own, which is the closest analogue when it has no demand row of its own.
    own = r if square else (sim.T @ r) * fid
    S = sorted(current.values(), key=lambda m: -own[m])
    spent = float(sum(cost[m] for m in S))
    greedy_val = objective(S, r, sim, fid)
    feasible = [m for m in range(n) if cost[m] <= budget]
    best_single = max(feasible, key=lambda m: objective([m], r, sim, fid), default=None)
    single_val = objective([best_single], r, sim, fid) if best_single is not None else 0.0
    if single_val > greedy_val:
        S, spent = [best_single], float(cost[best_single])
    return S, {"objective": max(greedy_val, single_val), "greedy_objective": greedy_val,
               "used_best_single": single_val > greedy_val, "cost": float(spent),
               "n_demand": int(len(r)), "n_candidates": int(n)}
