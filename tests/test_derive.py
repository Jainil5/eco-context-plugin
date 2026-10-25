"""Derived facts: generation guards, additive-only coverage, and the budget trade they enable.

Offline (WordEmbed), no network, no paid model. The properties pinned here:
  * a fact aggregates >= 2 mentions across >= 2 conversations, or a counting cue permits one conversation;
  * a derived fact covers NO span's demand - it is additive information, never a substitute for its evidence;
  * it carries its own demand row worth DERIVE_BETA of its covered spans' mass, at DERIVE_FID fidelity;
  * derive=False leaves the published objective untouched.
"""
from datetime import datetime

import numpy as np
import pytest

from core.context import Services
from core.data.longmemeval import Question, Turn
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient
from core.llm.cost import CostLogger, Pricing
from core.retrieval.spans import build_span_index
from systems.eco.derive import derive_facts
from systems.eco.objective import DERIVE_BETA, DERIVE_FID, build_objective
from systems.eco.spans import ECOspans

WORDS = ["cat", "miso", "pune", "gurgaon", "paris", "trip", "sister", "dog", "tea", "work", "live"]


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


def multi_session_turns():
    """'live in Pune' restated in three dated sessions; one unrelated cat conversation."""
    return [
        turn(("s0", 0), datetime(2023, 1, 10), "user", "I live in Pune with my sister and I work in Pune every day."),
        turn(("s0", 1), datetime(2023, 1, 10), "assistant", "Pune is busy, your commute sounds long."),
        turn(("s1", 0), datetime(2023, 4, 20), "user", "I still live in Pune, work nearby, no plans to move."),
        turn(("s1", 1), datetime(2023, 4, 20), "assistant", "Good that living in Pune suits you."),
        turn(("s2", 0), datetime(2023, 6, 2), "user", "I live in Pune now with two more roommates."),
        turn(("s2", 1), datetime(2023, 6, 2), "assistant", "Roommates in Pune sound fun."),
        turn(("s3", 0), datetime(2023, 7, 8), "user", "My cat Miso sleeps on the sofa all day."),
        turn(("s3", 1), datetime(2023, 7, 8), "assistant", "Cats nap a lot, Miso seems normal."),
    ]


def index_of(services, turns, qid="q1"):
    return build_span_index(turns, services, qid, min_tokens=4)


def pool(ix, query, k=100):
    return [i for i, _ in ix.fused(query, k)]


def test_derive_facts_aggregate_across_sessions(services):
    ix = index_of(services, multi_session_turns())
    query = "Where do I live?"
    cand = pool(ix, query)
    facts = derive_facts(ix, cand, query, datetime(2023, 8, 1))
    assert facts, "a facet repeated across three sessions must yield a fact"
    f = facts[0]
    assert "[derived]" in f.item.text
    assert f.n_sessions >= 2 and len(f.covered) >= 2
    assert not f.item.verbatim
    assert f.last_date > f.first_date


def test_single_mention_and_single_session_are_not_facts(services):
    ix = index_of(services, multi_session_turns())
    cand = pool(ix, "What does my cat do?")
    facts = derive_facts(ix, cand, "What does my cat do?", datetime(2023, 8, 1))
    assert all(f.n_sessions >= 2 for f in facts)          # counting cue absent: sessions >= 2 enforced


def test_counting_cue_allows_one_conversation_repeats(services):
    turns = [turn(("s0", 0), datetime(2023, 1, 10), "user", "I drink tea every morning, tea at lunch, tea at six."),
             turn(("s0", 1), datetime(2023, 1, 10), "assistant", "That is a lot of tea.")]
    ix = index_of(services, turns)
    cand = pool(ix, "How many times a day do I drink tea?")
    facts = derive_facts(ix, cand, "How many times a day do I drink tea?", datetime(2023, 8, 1))
    assert facts and "times" in facts[0].item.text


def test_derived_candidates_are_additive_in_the_objective(services):
    ix = index_of(services, multi_session_turns())
    query = "Where do I live?"
    cand = pool(ix, query)
    facts = derive_facts(ix, cand, query, datetime(2023, 8, 1))
    base = build_objective(ix, cand, 2.0, 0.7, query)
    obj = build_objective(ix, cand, 2.0, 0.7, query, derived=facts)
    n, k = len(cand), len(facts)
    assert obj.sim.shape == (n + k, n + k) and len(obj.r) == n + k
    assert np.allclose(obj.sim[:n, n:], 0.0)              # covers no span's demand: additive only
    for p, f in enumerate(facts):                        # cheaper than the union of its evidence, not one span
        assert obj.cost[n + p] < obj.cost[:n][f.covered].sum()
    assert np.allclose(obj.fid[n:], DERIVE_FID)
    for p, f in enumerate(facts):
        assert obj.r[n + p] == pytest.approx(DERIVE_BETA * base.r[f.covered].sum())


def test_derive_off_leaves_the_published_objective_untouched(services):
    ix = index_of(services, multi_session_turns())
    query = "Where do I live?"
    cand = pool(ix, query)
    a = build_objective(ix, cand, 2.0, 0.7, query)
    b = build_objective(ix, cand, 2.0, 0.7, query, derived=[])
    assert np.allclose(a.r, b.r) and np.allclose(a.sim, b.sim) and a.trace.get("n_derived") is None


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_long_date_lists_truncate_instead_of_crashing(services):
    """Regression: real dev data produced counting facts whose 8-date lists broke the render-cost guard."""
    from core.llm.tokenizer import get_tokenizer
    from systems.eco.derive import COUNT_CAP_TOKENS
    turns = [turn((f"s{i:02d}", 0), datetime(2023, m, d), "user",
                  f"On {m}/{d} I logged a trip to Paris with my sister.")
             for i, (m, d) in enumerate([(1, 5), (1, 20), (2, 3), (2, 18), (3, 7), (3, 22), (4, 9), (5, 1), (5, 17)])]
    ix = index_of(services, turns)
    query = "How many trips to Paris did I take?"
    cand = pool(ix, query)
    facts = derive_facts(ix, cand, query, datetime(2023, 8, 1))
    assert facts, "a facet repeated across nine dated sessions must yield a counting fact"
    for f in facts:
        assert get_tokenizer().count(f.item.text) + 1 <= COUNT_CAP_TOKENS
        assert f"[derived]" in f.item.text
        if "+ more" in f.item.text:
            assert f"mentioned {f.n_mentions} times" in f.item.text   # the count stays the true count


def test_eco_derive_selects_a_fact_only_when_budget_prices_it(services):
    q = Question("q1", "temporal-reasoning", "Where do I live?", datetime(2023, 8, 1), "Pune",
                 multi_session_turns(), ["s0:0"])
    eco = ECOspans(services, derive=True)
    mem = eco.build_memory(q)
    small = eco.build_context(q.question, mem, 150, q.question_date)
    big = eco.build_context(q.question, mem, 4000, q.question_date)
    assert "[derived]" in small.context                  # cheap relation beats displaced tail spans
    assert small.trace["n_derived_selected"] >= 1
    assert big.n_tokens <= 4000
    assert big.trace["n_derived_selected"] + sum(1 for _ in big.context.split("\n")) >= 2
