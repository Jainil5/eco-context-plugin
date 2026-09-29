import itertools

import numpy as np
import pytest

from systems.eco.select import objective, select


def instance(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 8))
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    sim = np.clip(x @ x.T, 0, None)
    return rng.random(n), sim, rng.choice([1.0, 0.8], size=n), rng.integers(1, 40, size=n).astype(float)


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("budget", [0, 5, 30, 100, 1e9])
def test_budget_never_exceeded(seed, budget):
    r, sim, fid, cost = instance(30, seed)
    S, info = select(r, sim, fid, cost, budget)
    assert cost[S].sum() <= budget and len(set(S)) == len(S)
    assert info["objective"] == pytest.approx(objective(S, r, sim, fid))


@pytest.mark.parametrize("seed", range(10))
def test_objective_monotone_and_submodular(seed):
    r, sim, fid, _ = instance(15, seed)
    rng = np.random.default_rng(seed)
    for _ in range(50):
        A = list(rng.choice(15, size=rng.integers(0, 6), replace=False))
        B = A + [m for m in rng.choice(15, size=4, replace=False) if m not in A]
        e = next(m for m in range(15) if m not in B)
        assert objective(B, r, sim, fid) >= objective(A, r, sim, fid) - 1e-12
        gain_a = objective(A + [e], r, sim, fid) - objective(A, r, sim, fid)
        gain_b = objective(B + [e], r, sim, fid) - objective(B, r, sim, fid)
        assert gain_a >= gain_b - 1e-12


@pytest.mark.parametrize("seed", range(10))
def test_near_optimal_on_small_instances(seed):
    r, sim, fid, cost = instance(10, seed)
    budget = 60
    best = max(objective(list(S), r, sim, fid) for k in range(11) for S in itertools.combinations(range(10), k)
               if cost[list(S)].sum() <= budget)
    S, _ = select(r, sim, fid, cost, budget)
    assert objective(S, r, sim, fid) >= 0.5 * (1 - 1 / np.e) * best - 1e-9


def test_prefers_cheap_equivalent_rendering():
    # item 1 is a cheap fact that covers item 0 (its source turn) almost fully
    r = np.array([1.0, 0.0])
    sim = np.array([[1.0, 1.0], [1.0, 1.0]])
    S, _ = select(r, sim, np.array([1.0, 0.8]), np.array([100.0, 10.0]), budget=50)
    assert S == [1]


@pytest.mark.parametrize("seed", range(20))
@pytest.mark.parametrize("budget", [5, 30, 100, 1e9])
def test_groups_keep_one_rendering_and_respect_budget(seed, budget):
    r, sim, fid, cost = instance(30, seed)
    group = np.arange(30) // 2  # pairs of renderings of the same unit
    S, info = select(r, sim, fid, cost, budget, group=group)
    assert cost[S].sum() <= budget
    assert len({group[m] for m in S}) == len(S)
    assert info["objective"] == pytest.approx(objective(S, r, sim, fid))


def test_upgrades_to_raw_rendering_when_budget_allows():
    # one unit, two renderings: a cheap lossy fact (fid .5) and the full raw turn (fid 1)
    r, sim = np.array([1.0, 1.0]), np.ones((2, 2))
    fid, cost, group = np.array([0.5, 1.0]), np.array([10.0, 100.0]), np.array([0, 0])
    assert select(r, sim, fid, cost, budget=50, group=group)[0] == [0]
    assert select(r, sim, fid, cost, budget=150, group=group)[0] == [1]
