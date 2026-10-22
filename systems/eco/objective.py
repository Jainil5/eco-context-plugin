"""What ECO actually optimises over sentence spans, as one testable object instead of inline arithmetic.

The selector (`systems/eco/select.py`) is generic and stays untouched. Everything that makes ECO more than
"diversified top-k" lives here, as four mechanisms on top of the same (r, sim, fid, cost) instance. Each is off by
default, so `build_objective(...)` with no flags reproduces the published span-level run exactly.

  supersession (gamma < 1). Two spans that restate the same fact at different dates are not interchangeable: the
      newer one is the current answer. Coverage is made asymmetric - an older span covers a newer one only at
      `gamma`, so leaving the newest statement out keeps real demand uncovered and greedy is forced to pay for it.
      Top-k and MMR cannot express this: both score items one at a time.

  one_per="session". At most one span per session may be selected (a second span of the same session is only an
      upgrade if it renders the same unit at higher fidelity, which two sentences of one turn are not). This is a
      property of the *set* - "the context must span several conversations" - which no per-item score can enforce.
      It is a hard cap, so it costs fidelity when one session legitimately holds most of the answer.

  session_weight > 0. The soft version of the same preference: one extra demand row per session, carrying a share
      of the total relevance mass split equally across rows, satisfied fully by that session's first span and worth
      nothing more after it. Greedy still buys a second span of a rich session when the marginal relevance gain
      beats spreading.

  facets > 0. The question is decomposed into its informative terms and each term becomes an extra demand row that
      cannot itself be selected. Covering two terms with one span is cheaper than covering them with two, so the
      objective now rewards spans that answer different parts of the question rather than the same part twice.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

from core.llm.tokenizer import get_tokenizer
from core.retrieval.bm25 import STOPWORDS, tokenize

FACET_MIN_DOCS = 2  # a term appearing in < 2 spans carries no discriminating information


@dataclass
class Objective:
    """Matrices for select(): r has len(rows) entries, sim is len(rows) x len(cand), fid/cost/group are len(cand)."""
    r: np.ndarray
    sim: np.ndarray
    fid: np.ndarray
    cost: np.ndarray
    group: np.ndarray | None
    cand: list[int]
    trace: dict = field(default_factory=dict)


def _dates(spans, turns) -> list:
    """Session date per candidate span; datetime.min when the haystack has no date."""
    return [(turns[s.turn_index].session_date or datetime.min) if s.turn_index < len(turns) else datetime.min
            for s in spans]


def _session_codes(spans, cand: list[int]) -> tuple[list[int], int]:
    """Stable 0-based id per session for the candidate spans (turn_id is "<session_id>:<turn>")."""
    codes, seen = [], {}
    for i in cand:
        sid = spans[i].turn_id.rsplit(":", 1)[0]
        seen.setdefault(sid, len(seen))
        codes.append(seen[sid])
    return codes, len(seen)


def facet_terms(query: str, index, cand: list[int], n_facets: int) -> list[str]:
    """Informative query terms, in order of appearance: not stopwords, and present in at least two candidates."""
    out: list[str] = []
    for tok in tokenize(query):
        if len(tok) < 3 or tok in STOPWORDS or tok in out:
            continue
        if np.count_nonzero(np.asarray(index.scores(tok))[cand] > 0) >= FACET_MIN_DOCS:
            out.append(tok)
        if len(out) == n_facets:
            break
    return out


def build_objective(index, cand: list[int], rel_tau: float | None, sim_floor: float, query: str,
                    supersede: float | None = None, one_per: str | None = None, facets: int = 0,
                    facet_weight: float = 1.0, session_weight: float = 0.0,
                    fused: list[tuple[int, float]] | None = None) -> Objective:
    """Assemble the ECO instance for one query over the candidate spans `cand` (best first).

    `fused` is the caller's already-computed fused ranking; it is only needed when rel_tau is None (score-proportional
    weights), and passing it avoids re-running BM25 + dense for every setting in a sweep.
    """
    spans, turns = index.spans, index.turns
    if rel_tau:
        r = np.exp(-np.arange(len(cand)) / rel_tau)          # rank-decayed: the fused order is the relevance
    else:
        pairs = fused if fused is not None else index.fused(query, len(cand))
        s = np.array([sc for _, sc in pairs])
        r = s / s.max() if s.size and s.max() > 0 else np.ones(len(cand))

    V = index.dense.vecs[cand]
    raw = np.clip(V @ V.T, -1.0, 1.0)
    sim = np.clip((raw - sim_floor) / (1.0 - sim_floor), 0.0, None)
    np.fill_diagonal(sim, 1.0)

    tok = get_tokenizer()
    cost = np.array([tok.count(spans[i].item.text) + 1 for i in cand], dtype=float)
    fid = np.ones(len(cand))
    # how many conversations the pool spans, whether or not a breadth mechanism is on: this is the number that says
    # whether the testbed can even show a mechanism working (scripts/mechanisms.py reports it per question).
    codes, n_sess = _session_codes(spans, cand)
    trace = {"superseded_pairs": 0, "n_candidate_sessions": n_sess, "facet_terms": []}

    if supersede is not None and supersede < 1.0:
        d = _dates([spans[i] for i in cand], turns)
        older = np.array([[1.0 if d[j] < d[i] else 0.0 for j in range(len(cand))] for i in range(len(cand))])
        restated = (raw >= sim_floor) & (older > 0)          # looks like a restatement AND is from an earlier day
        sim = np.where(restated, sim * supersede, sim)
        trace["superseded_pairs"] = int(restated.sum())

    group = np.array(codes) if one_per == "session" else None

    if session_weight > 0:
        # Soft breadth: one extra demand row per session in the pool. The first span of a session satisfies it
        # completely, so further spans of that session earn nothing from it - pressure to spread, not a cap.
        # `session_weight` is a share of the total relevance mass, not an absolute weight, so "breadth is worth
        # X of relevance" means the same thing whatever the pool size or tau. Entering an untouched session is
        # worth session_weight * mass / n_sessions, i.e. with weight 1.0 and a 40-session pool it is worth about
        # the same as an average candidate's own relevance - enough to flip a close call, not to override the top hits.
        rows = np.zeros((n_sess, len(cand)))
        for p, c in enumerate(codes):
            rows[c, p] = 1.0
        mass = float(r.sum()) * session_weight
        r = np.concatenate([r, np.full(n_sess, mass / n_sess)])
        sim = np.vstack([sim, rows])
        trace["session_mass_share"] = session_weight

    if facets:
        terms = facet_terms(query, index.bm25, cand, facets)
        if terms:
            rows = []
            for t in terms:
                sc = np.asarray(index.bm25.scores(t), dtype=np.float64)[cand]
                rows.append(np.clip(sc / max(sc.max(), 1e-9), 0.0, 1.0))
            r = np.concatenate([r, np.full(len(terms), facet_weight / len(terms))])
            sim = np.vstack([sim, np.array(rows)])
            trace["facet_terms"] = terms

    return Objective(r=r, sim=sim, fid=fid, cost=cost, group=group, cand=list(cand), trace=trace)
