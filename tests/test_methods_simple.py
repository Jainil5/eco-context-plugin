from datetime import datetime

import pytest

from core.data.longmemeval import Question, Turn
from core.eval.recall import evidence_recall
from core.llm.tokenizer import count_tokens
from systems.baselines.simple import BM25Turns, OracleEvidence, RecentWindow


def make_q():
    turns = []
    for s in range(6):
        for j, (role, text) in enumerate([("user", f"session {s} talk about topic{s} " * (5 + 10 * s)),
                                          ("assistant", "sure " * 20)]):
            turns.append(Turn(f"s{s}:{j}", f"s{s}", datetime(2023, 1, 1 + s), role, text, None))
    turns[2].text = "I adopted a cat named Miso"
    turns[2].has_answer = True
    return Question("q", "single-session-user", "What is my cat's name?", datetime(2023, 2, 1), "Miso",
                    turns, ["s1:0"])


METHODS = [RecentWindow(), BM25Turns(), OracleEvidence()]


@pytest.mark.parametrize("m", METHODS, ids=lambda m: m.name)
@pytest.mark.parametrize("budget", [5, 30, 100, 400, 5000])
def test_budget_never_exceeded_and_provenance_valid(m, budget):
    q = make_q()
    r = m.build_context(q.question, m.build_memory(q), budget, q.question_date)
    assert r.n_tokens == count_tokens(r.context) <= budget
    ids = {t.turn_id for t in q.haystack}
    assert set(r.source_turn_ids) <= ids
    assert all(sid in r.context for sid in r.source_turn_ids)


def test_bm25_and_oracle_find_evidence_recent_does_not():
    q = make_q()
    get = lambda m: m.build_context(q.question, m.build_memory(q), 60, q.question_date).source_turn_ids
    assert evidence_recall(get(BM25Turns()), q.evidence_turn_ids) == 1.0
    assert evidence_recall(get(OracleEvidence()), q.evidence_turn_ids) == 1.0
    assert evidence_recall(get(RecentWindow()), q.evidence_turn_ids) == 0.0


def test_context_is_chronological():
    q = make_q()
    ids = BM25Turns().build_context(q.question, BM25Turns().build_memory(q), 5000, None).source_turn_ids
    assert ids == [t.turn_id for t in q.haystack if t.turn_id in set(ids)]


def test_deterministic():
    q = make_q(); m = BM25Turns()
    a = m.build_context(q.question, m.build_memory(q), 200, q.question_date)
    b = m.build_context(q.question, m.build_memory(q), 200, q.question_date)
    assert a.context == b.context


def test_stemmed_tokenize_is_thread_safe():
    from concurrent.futures import ThreadPoolExecutor

    from core.retrieval.bm25 import tokenize

    texts = [f"traveling organizational generalizations relational {i} hopefully" * 20 for i in range(200)]
    expected = [tokenize(t, stem=True) for t in texts]
    with ThreadPoolExecutor(8) as ex:
        assert list(ex.map(lambda t: tokenize(t, stem=True), texts)) == expected
