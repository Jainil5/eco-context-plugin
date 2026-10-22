"""Objective mechanisms: supersession, one-per-session, facet coverage. Offline, no network, no paid model.

Each test pins a property the mechanism is supposed to have, plus the guard that turning everything off leaves the
published span-level behaviour untouched (that guard is what makes the mechanism comparison honest).
"""
from datetime import datetime

import numpy as np
import pytest

from core.context import Services
from core.data.longmemeval import Question, Turn
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient
from core.llm.cost import CostLogger, Pricing
from core.llm.tokenizer import get_tokenizer
from core.retrieval.spans import build_span_index
from systems.eco.objective import build_objective, facet_terms
from systems.eco.select import select
from systems.eco.spans import ECOspans

WORDS = ["cat", "miso", "pune", "gurgaon", "paris", "trip", "sister", "dog", "tea", "work"]


class WordEmbed:
    model = "word-bag"

    def embed_documents(self, texts):
        return [[float(max(t.lower().count(w), 0.01)) for w in WORDS] for t in texts]


@pytest.fixture
def services(tmp_path):
    log = CostLogger(tmp_path / "costs.jsonl", Pricing({}))
    return Services(llms={}, embedder=EmbeddingClient(WordEmbed(), DiskCache(tmp_path / "c.sqlite"), log))


def turn(sid_idx, date, role, text):
    sid, idx = sid_idx
    return Turn(f"{sid}:{idx}", sid, date, role, text, None)


OLD, NEW = datetime(2023, 1, 10), datetime(2023, 4, 20)


def supersede_turns():
    """One fact, restated verbatim in a later session. BM25 puts the older copy first (tie broken by document
    order), so a plain coverage objective is happy to keep the stale one; the newest statement is the answer."""
    same = "I live in Pune with my sister and I work in Pune every single day."
    return [
        turn(("s0", 0), OLD, "user", same),
        turn(("s0", 1), OLD, "assistant", "Pune is a busy city, your commute sounds long."),
        turn(("s1", 0), NEW, "user", same),
        turn(("s1", 1), NEW, "assistant", "Still Pune then, good to hear."),
        turn(("s2", 0), NEW, "user", "My cat Miso sleeps on the sofa all day."),
        turn(("s2", 1), NEW, "assistant", "Cats nap a lot, Miso seems normal."),
    ]


def index_of(services, turns, qid="q1"):
    return build_span_index(turns, services, qid, min_tokens=4)


def pool(ix, query, k=100):
    return [i for i, _ in ix.fused(query, k)]


# --- guard: mechanisms off == the published span-level arithmetic -------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_plain_objective_matches_the_published_inline_formulas(services):
    ix = index_of(services, supersede_turns())
    query = "Where do I live?"
    cand = pool(ix, query)
    obj = build_objective(ix, cand, 2.0, 0.7, query)
    r = np.exp(-np.arange(len(cand)) / 2.0)
    V = ix.dense.vecs[cand]
    sim = np.clip((V @ V.T - 0.7) / 0.3, 0.0, None)
    np.fill_diagonal(sim, 1.0)
    cost = np.array([get_tokenizer().count(ix.spans[i].item.text) + 1 for i in cand], dtype=float)
    assert np.allclose(obj.r, r) and np.allclose(obj.sim, sim) and np.allclose(obj.cost, cost)
    assert np.allclose(obj.fid, 1.0) and obj.group is None
    assert obj.trace["superseded_pairs"] == 0 and obj.trace["facet_terms"] == []


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_off_variants_select_identically(services):
    """eco_spans and a variant built with every mechanism off must produce the same context."""
    q = Question("q1", "single-session-user", "Where do I live?", NEW, "Gurgaon", supersede_turns(), ["s1:0"])
    a = ECOspans(services)
    b = ECOspans(services, supersede=None, one_per=None, facets=0)
    assert a.build_context(q.question, a.build_memory(q), 60, q.question_date).context == \
           b.build_context(q.question, b.build_memory(q), 60, q.question_date).context


# --- supersession ------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_supersession_discounts_only_the_older_direction(services):
    ix = index_of(services, supersede_turns())
    cand = pool(ix, "Where do I live?")
    plain = build_objective(ix, cand, 2.0, 0.7, "Where do I live?")
    sup = build_objective(ix, cand, 2.0, 0.7, "Where do I live?", supersede=0.5)
    dates = {i: ix.turns[ix.spans[i].turn_index].session_date for i in cand}
    changed = [(i, m) for i in range(len(cand)) for m in range(len(cand))
               if dates[cand[m]] < dates[cand[i]]]
    assert changed, "fixture has no older candidate"
    for i, m in changed:
        if plain.sim[i, m] > 0:                      # they look like restatements of each other
            assert sup.sim[i, m] == pytest.approx(plain.sim[i, m] * 0.5)
        assert sup.sim[m, i] == plain.sim[m, i]      # a newer span still covers an older demand fully
    assert sup.trace["superseded_pairs"] > 0


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_supersession_buys_the_newest_statement(services):
    """Budget 40 buys exactly one span: plain coverage takes the older copy of the fact, supersession the newer."""
    query = "Where do I live?"
    ix = index_of(services, supersede_turns())
    cand = pool(ix, query)
    newest = {p for p, i in enumerate(cand) if ix.spans[i].turn_id.startswith("s1")}  # positions, not span ids
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    sup = build_objective(ix, cand, 2.0, 0.7, query, supersede=0.1)
    S_plain, _ = select(plain.r, plain.sim, plain.fid, plain.cost, 40)
    S_sup, _ = select(sup.r, sup.sim, sup.fid, sup.cost, 40)
    assert not (set(S_plain) & newest), "fixture must be hard: plain coverage keeps the stale statement"
    assert set(S_sup) & newest, "supersession should pay for the current statement"


# --- one span per session ----------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_one_per_session_forces_breadth(services):
    query = "Where do I live and what is the weather like?"
    ix = index_of(services, supersede_turns())
    cand = pool(ix, query)
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    clos = build_objective(ix, cand, 2.0, 0.7, query, one_per="session")
    assert clos.group is not None and set(clos.group) == {0, 1, 2} and clos.trace["n_candidate_sessions"] == 3

    def sessions(S):
        return {ix.spans[cand[j]].turn_id.rsplit(":", 1)[0] for j in S}

    for budget in (25, 35, 45, 60):
        Sp, _ = select(plain.r, plain.sim, plain.fid, plain.cost, budget)
        Sc, _ = select(clos.r, clos.sim, clos.fid, clos.cost, budget, group=clos.group)
        assert len(sessions(Sc)) == len(Sc), "one_per=session must never take two spans from one session"
        assert len(sessions(Sc)) >= len(sessions(Sp)), "the constraint must not cover fewer sessions"


# --- soft breadth (session_weight) --------------------------------------------------------

@pytest.mark.filterwarnings("ignore::UserWarning")
def test_session_rows_are_session_indicators_scaled_by_mass_share(services):
    query = "Where do I live and what is my cat called?"
    ix = index_of(services, supersede_turns())
    cand = pool(ix, query)
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    obj = build_objective(ix, cand, 2.0, 0.7, query, session_weight=1.0)
    n, k = len(cand), obj.trace["n_candidate_sessions"]
    assert k == 3 and obj.trace["session_mass_share"] == 1.0
    assert obj.sim.shape == (n + k, n) and obj.r.shape == (n + k,)
    rows = obj.sim[n:]
    assert np.allclose(rows.sum(axis=0), 1.0), "every candidate belongs to exactly one session row"
    for row in rows:
        assert set(np.unique(row)) <= {0.0, 1.0}
    # the extra mass is `session_weight` times the relevance mass, split equally over the sessions
    assert np.allclose(obj.r[n:], float(plain.r.sum()) * 1.0 / k)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_second_span_of_a_visited_session_earns_no_breadth(services):
    """Marginal breadth gain: zero inside a session the context already touches, full weight in one it does not."""
    query = "Where do I live and what is my cat called?"
    ix = index_of(services, supersede_turns())
    cand = pool(ix, query)
    obj = build_objective(ix, cand, 2.0, 0.7, query, session_weight=1.0)
    n, rows, w = len(cand), obj.sim[len(cand):], obj.r[len(cand):]
    sess = [ix.spans[i].turn_id.rsplit(":", 1)[0] for i in cand]
    first = int(obj.cost.argmin())                            # a span cheap enough to assume it is already in
    S = [first]
    cover = rows[:, S].max(axis=1)
    for j in range(n):
        gain = float(((rows[:, j] - cover) * w).clip(min=0).sum())
        if sess[j] == sess[cand[first]]:
            assert gain == 0.0, "a second span of a covered session must earn no breadth"
        else:
            assert gain == pytest.approx(float(w[rows[:, j].argmax()])), "an untouched session earns its full row"


def test_soft_breadth_buys_a_session_by_trading_fidelity():
    """Pure selector case, because coverage already diversifies on its own: three spans, two sessions.

    Span 1 is mildly redundant with span 0 (same session) and still worth more than span 2 (other session), so plain
    coverage takes both spans of s0. The session rows are worth more than that margin, so the set spreads - and it
    scores *lower* under the plain objective. That is the trade the knob makes; it is not a free win.
    """
    r = np.array([1.0, 0.607, 0.368])
    sim = np.array([[1.0, 0.2, 0.0], [0.2, 1.0, 0.0], [0.0, 0.0, 1.0]])
    fid, cost = np.ones(3), np.full(3, 30.0)
    Sp, _ = select(r, sim, fid, cost, budget=60)
    assert Sp == [0, 1], "fixture must be a case coverage answers by concentrating in one session"

    rows = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])          # demand: one row per session
    mass = float(r.sum()) * 1.0                                  # session_weight = 1.0
    rb = np.concatenate([r, np.full(2, mass / 2)])
    Sb, info = select(rb, np.vstack([sim, rows]), fid, cost, budget=60)
    assert sorted(Sb) == [0, 2], "breadth must take the untouched session instead of the second span of s0"
    assert max(Sb) < 3 and info["n_demand"] == 5 and info["n_candidates"] == 3

    def coverage_only(S):
        return float((r * sim[:, S].max(axis=1)).sum())

    assert coverage_only(Sb) < coverage_only(Sp), "the spread set must cost coverage - that is the knob's price"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_zero_session_weight_changes_nothing(services):
    """session_weight=0 must leave the matrices at the published square shape (no silent extra demand rows)."""
    query = "Where do I live?"
    ix = index_of(services, supersede_turns())
    cand = pool(ix, query)
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    off = build_objective(ix, cand, 2.0, 0.7, query, session_weight=0.0)
    assert off.sim.shape == plain.sim.shape and np.allclose(off.r, plain.r) and np.allclose(off.sim, plain.sim)
    assert "session_mass_share" not in off.trace


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_breadth_cannot_create_sessions_the_pool_does_not_have(services):
    """The oracle view is a single evidence session, and no breadth knob can spread a context over sessions that
    are not in the pool. This is why the mechanism has to be evaluated in view=full, not here."""
    query = "Where do I live?"
    pool_turns = [t for t in supersede_turns() if t.session_id == "s1"]
    ix = index_of(services, pool_turns)
    cand = pool(ix, query)
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    soft = build_objective(ix, cand, 2.0, 0.7, query, session_weight=4.0)
    clos = build_objective(ix, cand, 2.0, 0.7, query, one_per="session")
    assert soft.trace["n_candidate_sessions"] == clos.trace["n_candidate_sessions"] == 1
    for obj, group in ((plain, None), (soft, None), (clos, clos.group)):
        S, _ = select(obj.r, obj.sim, obj.fid, obj.cost, 120, group=group)
        assert {ix.spans[cand[j]].turn_id.rsplit(":", 1)[0] for j in S} == {"s1"}


# --- facets -------------------------------------------------------------------------------

def facet_turns():
    """Eight spans so a term in two of them still has positive BM25 idf (in a 4-doc corpus it would be zero)."""
    return [
        turn(("s0", 0), OLD, "user", "My cat Miso sleeps on the sofa all day."),
        turn(("s0", 1), OLD, "assistant", "Cats sleep a lot, but your cat Miso seems content."),
        turn(("s1", 0), NEW, "user", "My trip to Paris is in June with my sister."),
        turn(("s1", 1), NEW, "assistant", "Paris in June is wonderful for a trip."),
        turn(("s2", 0), NEW, "user", "I drink tea at work every afternoon."),
        turn(("s2", 1), NEW, "assistant", "Tea at work is a fine habit."),
        turn(("s3", 0), NEW, "user", "My dog stays at the office near the park."),
        turn(("s3", 1), NEW, "assistant", "Dogs need long walks after work."),
    ]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_facet_terms_are_informative(services):
    ix = index_of(services, facet_turns())
    cand = pool(ix, "What is my cat called and where is my trip?")
    terms = facet_terms(query="What is my cat called and where is my trip?", index=ix.bm25, cand=cand, n_facets=3)
    assert "cat" in terms and "trip" in terms       # both appear in >= 2 candidates
    assert "and" not in terms and "what" not in terms  # stopwords never become facets
    assert len(terms) <= 3


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_facets_add_demand_rows_that_cannot_be_selected(services):
    query = "What is my cat called and where is my trip?"
    ix = index_of(services, facet_turns())
    cand = pool(ix, query)
    obj = build_objective(ix, cand, 2.0, 0.7, query, facets=2)
    n = len(cand)
    assert obj.r.shape == (n + len(obj.trace["facet_terms"]),)
    assert obj.sim.shape == (n + len(obj.trace["facet_terms"]), n)
    S, info = select(obj.r, obj.sim, obj.fid, obj.cost, 40)
    assert max(S) < n, "a facet row must never end up in the context"
    assert info["n_demand"] == n + len(obj.trace["facet_terms"]) and info["n_candidates"] == n


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_facet_coverage_spans_both_parts_of_the_question(services):
    query = "What is my cat called and where is my trip?"
    ix = index_of(services, facet_turns())
    cand = pool(ix, query)
    plain = build_objective(ix, cand, 2.0, 0.7, query)
    fac = build_objective(ix, cand, 2.0, 0.7, query, facets=2)
    terms = fac.trace["facet_terms"]
    assert len(terms) == 2, f"fixture should yield two facets, got {terms}"
    rows = fac.sim[len(cand):]                       # facet demand rows
    budget = 70  # room for two spans

    def covered(S):
        return sum(1 for row in rows if max((row[j] * fac.fid[j] for j in S), default=0.0) > 0.5)

    Sp, _ = select(plain.r, plain.sim, plain.fid, plain.cost, budget)
    Sf, _ = select(fac.r, fac.sim, fac.fid, fac.cost, budget)
    assert covered(Sf) == 2, "with facets on both parts of the question must be answered"
    assert covered(Sp) <= covered(Sf)


# --- solver: rectangular demand -----------------------------------------------------------

def test_rectangular_select_is_monotone_and_budget_feasible():
    rng = np.random.default_rng(0)
    n, d = 6, 9
    V = rng.normal(size=(n, 4))
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    sim = np.vstack([np.clip(V @ V.T, 0, 1), np.clip(rng.normal(size=(d - n, n)), 0, 1)])
    r = np.ones(d)
    fid, cost = np.ones(n), np.arange(1, n + 1, dtype=float)
    S, info = select(r, sim, fid, cost, budget=7)
    assert sum(cost[S]) <= 7 and info["n_demand"] == d and info["n_candidates"] == n
    bigger, _ = select(r, sim, fid, cost, budget=21)
    assert len(bigger) >= len(S)


def test_square_select_order_is_unchanged():
    """The published path: with n x n demand the selected set is ordered by each item's own relevance."""
    r = np.array([1.0, 0.5, 0.25])
    sim = np.eye(3)
    S, info = select(r, sim, np.ones(3), np.ones(3), budget=2)
    assert S == [0, 1] and info["objective"] == 1.5
