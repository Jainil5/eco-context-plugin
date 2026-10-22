from datetime import datetime

import numpy as np
import pytest

from core.context import Services, span_units
from core.data.longmemeval import Question, Turn
from core.eval.recall import content_recall, evidence_recall
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient
from core.llm.cost import CostLogger, Pricing
from core.llm.tokenizer import count_tokens
from core.retrieval.mmr import mmr_order
from systems.baselines.span import SpanHybridRRF, SpanMMR
from systems.eco.spans import ECOspans

WORDS = ["cat", "miso", "name", "paris", "trip", "tea", "work", "dog", "park", "sister"]


class WordEmbed:
    """One dimension per word: a span about Miso is orthogonal to a span about Paris, so the coverage
    objective has something real to cover."""
    model = "word-bag"

    def embed_documents(self, texts):
        return [[float(max(t.lower().count(w), 0.01)) for w in WORDS] for t in texts]


def q_turns(*items):
    d = datetime(2023, 5, 1)
    return [Turn(f"s{t // 3}:{t % 3}", f"s{t // 3}", d, role, text, None) for t, (role, text) in enumerate(items)]


def make_q():
    turns = q_turns(
        ("user", "My cat is named Miso and she is eight years old."),
        ("assistant", "Miso is a lovely name for a cat."),
        ("user", "I am planning a trip to Paris in June with my sister."),
        ("assistant", "Paris is wonderful in June, the museums are open late."),
        ("user", "I drink tea at work every afternoon to stay focused."),
        ("assistant", "Tea is a good choice for the afternoon."),
        ("user", "My sister has a dog called Park near the office."),
        ("assistant", "Dogs need exercise near the office."),
    )
    return Question("q1", "multi-session", "What is my cat's name and where am I travelling?",
                    datetime(2023, 6, 1), "Miso, Paris", turns, ["s0:0", "s1:0"],
                    evidence_session_ids=["s0", "s1"])


@pytest.fixture
def services(tmp_path):
    log = CostLogger(tmp_path / "costs.jsonl", Pricing({}))
    return Services(llms={}, embedder=EmbeddingClient(WordEmbed(), DiskCache(tmp_path / "c.sqlite"), log))


# --- metric -------------------------------------------------------------------

def test_sentences_are_contiguous_substrings():
    from core.text import sentences
    text = "First stop. Second one is longer, yes! Third."
    for s in sentences(text, min_chars=0):
        assert s in text


def test_content_recall_whole_turn_equals_turn_recall():
    turns = make_q().haystack
    gold = [t.text for t in turns if t.turn_id == "s0:0"]
    ctx = turns[0].text
    assert content_recall(ctx, gold) == 1.0
    assert content_recall("nothing here", gold) == 0.0
    assert content_recall(ctx, []) is None


def test_content_recall_gives_partial_credit():
    gold = ["The cat named Miso is eight years old and very calm. "
            "She sits by the living room window every single afternoon."]
    assert content_recall("The cat named Miso is eight years old and very calm.", gold) == pytest.approx(0.5)
    assert content_recall("She sits by the living room window every single afternoon.", gold) == pytest.approx(0.5)


def test_evidence_recall_unchanged():
    assert evidence_recall(["a"], ["a", "b"]) == 0.5
    assert evidence_recall(["a"], []) is None


# --- span renderings ----------------------------------------------------------

def test_span_units_header_only_once_and_not_verbatim():
    from core.llm.tokenizer import get_tokenizer
    spans = span_units(make_q().haystack, get_tokenizer(), min_tokens=8)
    first_turn = [s for s in spans if s.turn_index == 0]
    assert all(s.item.verbatim is False for s in spans)
    assert first_turn[0].item.text.startswith("[2023-05-01 | s0:0] user: ")
    if len(first_turn) > 1:  # later spans cite the turn id without repaying the date and role
        assert first_turn[1].item.text.startswith("[s0:0] ")
        assert count_tokens(first_turn[1].item.text) < count_tokens(first_turn[0].item.text) + 10
    assert all(s.turn_id in s.item.source_turn_ids for s in spans)


def test_span_units_keep_short_sentences_attached():
    from core.llm.tokenizer import get_tokenizer
    turns = q_turns(("user", "Yes. I went to Paris in June last year."))
    spans = span_units(turns, get_tokenizer(), min_tokens=8)
    assert len(spans) == 1 and "Yes." in spans[0].text


# --- selectors over spans -----------------------------------------------------

@pytest.mark.parametrize("name,cls", [("span_hybrid", SpanHybridRRF), ("span_mmr", SpanMMR),
                                      ("eco_spans", ECOspans)])
@pytest.mark.parametrize("budget", [0, 20, 60, 128, 1000])
def test_span_systems_respect_budget_and_cite_only_evidence_turns(services, name, cls, budget):
    q = make_q()
    s = cls(services) if name != "span_mmr" else cls(services)
    r = s.build_context(q.question, s.build_memory(q), budget, q.question_date)
    assert r.n_tokens == count_tokens(r.context) <= budget
    assert not r.verbatim_turn_ids  # sentences are not whole turns
    assert set(r.source_turn_ids) <= {t.turn_id for t in q.haystack}


def test_answer_in_context_credits_a_paraphrase():
    from core.eval.recall import answer_in_context
    assert answer_in_context("[2023-05-01 | s0:1] assistant: Miso is a lovely cat name", "Miso") == 1.0
    assert answer_in_context("The trip is to Paris, in June.", "45 minutes each way") == 0.0
    assert answer_in_context("She studies Business Administration at Drexel.", "Business Administration") == 1.0
    assert answer_in_context("anything", "You did not mention this information.") is None  # abstention
    assert answer_in_context("anything", None) is None


def test_mmr_penalises_an_exact_duplicate():
    rel = np.array([0.9, 0.88, 0.5])
    V = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])  # 1 repeats 0 exactly
    order = mmr_order(rel, V, lam=0.7)
    assert order[0] == 0
    assert order[1] == 2  # the distinct item outranks the duplicate
    assert mmr_order(rel, V, lam=1.0)[1] == 1  # lambda=1 is pure relevance: the duplicate wins
