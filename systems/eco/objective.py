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
DERIVE_FID = 0.9    # a derived fact renders its aggregated spans at a loss: prefer the raw text when both fit
DERIVE_BETA = 0.25  # the fact itself (a count, a recency) is new demand worth a share of its evidence's mass
AUTO_TAU_RULES = ("fit", "elbow")
AUTO_TAU_CLIP = (1.0, 20.0)


def _fit_tau(s: np.ndarray) -> float | None:
    """Least-squares exponential decay rate of the top-20 candidates' EXCESS score over the pool median.

    Fitting the raw cosine score is degenerate: dense scores live in a narrow band (~0.3-0.9), so ln(score) barely
    falls with rank and -1/slope overflows the clip on every question. Subtracting the pool median gives a quantity
    that does decay over an order of magnitude - a sharp retrieval peak is a fast decay (small tau), a flat pool a
    slow one (large tau). None when fewer than 5 points sit above the median or the excess does not decay.
    """
    if len(s) < 2:
        return None
    med = float(np.median(s))
    k = min(len(s), 20)
    x, e = np.arange(1, k + 1, dtype=float), s[:k] - med
    m = e > 1e-9
    x, e = x[m], e[m]
    if len(x) < 5:
        return None
    y = np.log(e)
    xc = x - x.mean()
    denom = float((xc * xc).sum())
    if denom == 0.0:
        return None
    slope = float((xc * (y - y.mean())).sum() / denom)
    return None if slope >= -1e-6 else -1.0 / slope


def _elbow_tau(s: np.ndarray) -> float | None:
    """How many candidates sit above the midpoint between the best dense score and the pool median:
    a wide plateau (weak retriever separation) gives a large tau, a sharp peak a small one."""
    if len(s) < 2:
        return None
    thr = (float(s[0]) + float(np.median(s))) / 2.0
    return float(np.count_nonzero(s > thr))


def _auto_tau(rule: str, s: np.ndarray) -> float | None:
    tau = _fit_tau(s) if rule == "fit" else _elbow_tau(s)
    if tau is None:
        return None
    lo, hi = AUTO_TAU_CLIP
    return min(max(tau, lo), hi)


def _dense_scores_in_fused_order(index, query: str, cand: list[int]) -> np.ndarray:
    """Cosine similarity of each fused-ordered candidate against the query (candidates are already best-first)."""
    qv = index.dense.query_vec(query, index.qid)
    return np.clip(index.dense.vecs[cand] @ qv, 0.0, 1.0)


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
                    fused: list[tuple[int, float]] | None = None,
                    auto_tau: str | None = None, derived: list | None = None) -> Objective:
    """Assemble the ECO instance for one query over the candidate spans `cand` (best first).

    `fused` is the caller's already-computed fused ranking; it is only needed when rel_tau is None (score-proportional
    weights), and passing it avoids re-running BM25 + dense for every setting in a sweep.

    `auto_tau` ("fit" | "elbow") replaces the frozen rank-decay constant with a per-question one, estimated from the
    decay of the candidates' dense scores; rel_tau then serves as the fallback when the estimate is degenerate.

    `derived` (systems/eco/derive.py) appends symbolic aggregation facts as extra selectable candidates: they cover
    the spans aggregated into them at DERIVE_FID fidelity for a fraction of the tokens, and carry their own demand
    row worth DERIVE_BETA of that mass, so selecting one needs no LLM and is never free - it must beat the raw
    evidence on objective gain per token, which is what the same budget constraint prices.
    """
    spans, turns = index.spans, index.turns
    tau_used = None
    if auto_tau:
        s = _dense_scores_in_fused_order(index, query, cand)
        tau_used = _auto_tau(auto_tau, s)
        if tau_used is not None:
            r = np.exp(-np.arange(len(cand)) / tau_used)
        else:
            tau_used = rel_tau if rel_tau else 2.0
            r = np.exp(-np.arange(len(cand)) / tau_used)
    elif rel_tau:
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
    if auto_tau:
        trace.update({"auto_tau_rule": auto_tau, "tau_used": float(tau_used)})

    if supersede is not None and supersede < 1.0:
        d = _dates([spans[i] for i in cand], turns)
        older = np.array([[1.0 if d[j] < d[i] else 0.0 for j in range(len(cand))] for i in range(len(cand))])
        restated = (raw >= sim_floor) & (older > 0)          # looks like a restatement AND is from an earlier day
        sim = np.where(restated, sim * supersede, sim)
        trace["superseded_pairs"] = int(restated.sum())

    k = len(derived or [])
    if k:
        # Derived facts are ADDITIVE candidates: they cover no span's demand (a fact about evidence is not the
        # evidence), only their own row, so greedy takes one exactly when its relation is worth more objective
        # gain per token than the raw spans it displaces. Their columns stay zero for every span row below.
        sim = np.hstack([sim, np.zeros((len(cand), k))])
        cost = np.concatenate([cost, [tok.count(d.item.text) + 1 for d in derived]])
        fid = np.concatenate([fid, np.full(k, DERIVE_FID)])
        trace["n_derived"] = k

    group = np.array(codes) if one_per == "session" else None
    if group is not None and k:
        group = np.concatenate([group, np.arange(n_sess, n_sess + k)])  # each fact is its own group, uncapped

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
        mass = float(r[:len(cand)].sum()) * session_weight
        r = np.concatenate([r, np.full(n_sess, mass / n_sess)])
        sim = np.vstack([sim, np.hstack([rows, np.zeros((n_sess, k))]) if k else rows])
        trace["session_mass_share"] = session_weight

    if facets:
        terms = facet_terms(query, index.bm25, cand, facets)
        if terms:
            rows = []
            for t in terms:
                sc = np.asarray(index.bm25.scores(t), dtype=np.float64)[cand]
                rows.append(np.clip(sc / max(sc.max(), 1e-9), 0.0, 1.0))
            r = np.concatenate([r, np.full(len(terms), facet_weight / len(terms))])
            facet_mat = np.array(rows)
            sim = np.vstack([sim, np.hstack([facet_mat, np.zeros((len(terms), k))]) if k else facet_mat])
            trace["facet_terms"] = terms

    if k:
        # Each fact's own demand row: reachable only by selecting the fact. Its mass is a share of the relevance
        # of the spans it aggregates, so a fact about the tail is cheap demand and a fact about the top is not.
        r_span = r[:len(cand)]
        drows = np.zeros((k, len(cand) + k))
        for p, d in enumerate(derived):
            drows[p, len(cand) + p] = 1.0
        sim = np.vstack([sim, drows])
        r = np.concatenate([r, np.array([DERIVE_BETA * float(r_span[d.covered].sum()) for d in derived])])

    return Objective(r=r, sim=sim, fid=fid, cost=cost, group=group, cand=list(cand), trace=trace)
