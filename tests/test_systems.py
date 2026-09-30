import json
from datetime import datetime

import numpy as np
import pytest

from core.context import Services
from core.data.longmemeval import Question, Turn
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient
from core.llm.cost import CostLogger, Pricing
from core.llm.tokenizer import count_tokens
from systems import LLM_FREE, build_system


class BagEmbed:
    """Deterministic bag-of-words embedding over a tiny vocabulary (no network)."""
    model = "bag"
    VOCAB = ["cat", "miso", "name", "tea", "paris", "trip", "dog", "work", "hello"]

    def embed_documents(self, texts):
        return [[t.lower().count(w) + 0.01 for w in self.VOCAB] for t in texts]


def make_q():
    d = datetime(2023, 5, 1)
    texts = [("user", "hello there"), ("assistant", "hi, how can I help"),
             ("user", "my cat is named Miso"), ("assistant", "Miso is a lovely cat name"),
             ("user", "planning a trip to Paris"), ("assistant", "Paris is great in spring"),
             ("user", "I drink tea at work"), ("assistant", "tea is nice"),
             ("user", "my cat Miso loves tea"), ("assistant", "cute")]
    turns = [Turn(f"s{i // 4}:{i % 4}", f"s{i // 4}", d, role, text, None) for i, (role, text) in enumerate(texts)]
    turns[2].has_answer = True
    return Question("q1", "single-session-user", "What is my cat's name?", datetime(2023, 6, 1), "Miso",
                    turns, ["s0:2"], evidence_session_ids=["s0"])


@pytest.fixture
def services(tmp_path):
    log = CostLogger(tmp_path / "costs.jsonl", Pricing({}))
    emb = EmbeddingClient(BagEmbed(), DiskCache(tmp_path / "c.sqlite"), log)
    return Services(llms={}, embedder=emb)  # no chat model: LLM-free systems must not need one


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", sorted(LLM_FREE))
@pytest.mark.parametrize("budget", [0, 15, 40, 1000])
def test_llm_free_systems_respect_budget_and_provenance(services, name, budget):
    q = make_q()
    s = build_system(name, services)
    r = s.build_context(q.question, s.build_memory(q), budget, q.question_date)
    assert r.n_tokens == count_tokens(r.context) <= budget
    assert set(r.verbatim_turn_ids) <= set(r.source_turn_ids) <= {t.turn_id for t in q.haystack}
    if budget == 1000:  # everything fits
        assert "s0:2" in r.verbatim_turn_ids
        if name != "eco":  # eco skips zero-gain turns (exact duplicates under the embedding)
            assert set(r.verbatim_turn_ids) == {t.turn_id for t in q.haystack}


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("name", ["hybrid", "mmr", "oracle", "eco"])
def test_one_turn_budget_picks_the_evidence(services, name):
    q = make_q()
    s = build_system(name, services)
    r = s.build_context(q.question, s.build_memory(q), 40, q.question_date)  # room for about one turn
    assert r.verbatim_turn_ids == ["s0:2"]


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_mmr_skips_near_duplicates(services):
    q = make_q()
    dup = Turn("s0:1", "s0", q.haystack[0].session_date, "user", "my cat is named Miso", None)
    q.haystack[1] = dup  # identical twin of the evidence turn
    s = build_system("mmr", services)
    order = [t.turn_id for t in s.ranked(q.question, s.build_memory(q))]
    assert set(order[:2]) != {"s0:1", "s0:2"}


def test_unknown_system():
    with pytest.raises(KeyError):
        build_system("nope", Services())


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_runner_recall_mode_needs_no_chat_model(tmp_path):
    from core.runner import Runner
    cfg = {"paths": {"cache": str(tmp_path / "c.sqlite")}, "pricing": {}}
    (tmp_path / "run" / "traces").mkdir(parents=True)
    Runner(cfg, tmp_path / "run", None, None, BagEmbed(), eval_mode="recall").run(
        [make_q()], ["bm25", "eco"], [25, 1000], workers=2, progress=lambda *_: None)
    assert not (tmp_path / "run" / "errors.jsonl").exists()
    rows = [json.loads(l) for l in open(tmp_path / "run" / "predictions.jsonl")]
    assert len(rows) == 4 and all("correct" not in r for r in rows)
    assert all(r["verbatim_recall"] == 1.0 for r in rows if r["budget"] == 1000)
    assert all(0.0 <= r["verbatim_recall"] <= r["evidence_recall"] <= 1.0 for r in rows)
    assert np.all([r["n_tokens"] <= r["budget"] for r in rows])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_tables_built_from_run(tmp_path):
    import csv
    from core.eval.table import build_tables
    from core.runner import Runner
    cfg = {"paths": {"cache": str(tmp_path / "c.sqlite")}, "pricing": {}}
    run = tmp_path / "20260101-000000_compare_x"
    (run / "traces").mkdir(parents=True)
    Runner(cfg, run, None, None, BagEmbed(), eval_mode="recall").run(
        [make_q()], ["bm25", "eco"], [60, 1000], workers=1, progress=lambda *_: None)
    full, summary = build_tables(run, tmp_path / "tables")
    rows = list(csv.DictReader(open(summary)))
    assert {(r["system"], r["budget"]) for r in rows} == {(s, b) for s in ("bm25", "eco") for b in ("60", "1000")}
    assert all(r["qtype"] == "ALL" and r["accuracy"] == "" for r in rows)
    assert float(next(r for r in rows if r["system"] == "bm25" and r["budget"] == "1000")["verbatim_recall"]) == 1.0
    assert len(list(csv.DictReader(open(full)))) == 8  # ALL + one qtype, per system x budget


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_untuned_eco_prefers_generic_turn_at_one_turn_budget(services):
    """Regression record: with flat RRF relevance and raw cosine, ECO picked a short generic turn ('cute')."""
    from systems.eco.method import ECO
    q = make_q()
    s = ECO(services, rel_tau=None, sim_floor=0.0)
    r = s.build_context(q.question, s.build_memory(q), 40, q.question_date)
    assert r.verbatim_turn_ids == ["s2:1"]
