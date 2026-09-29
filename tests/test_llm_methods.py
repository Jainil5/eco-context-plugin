import json
import re
from datetime import datetime

import numpy as np
import pytest
from langchain_core.messages import AIMessage

from core.data.longmemeval import Question, Turn
from systems.eco.method import ECO
from core.eval.recall import evidence_recall
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient, LLMClient
from core.llm.cost import CostLogger, Pricing
from core.llm.tokenizer import count_tokens
from core.context import Services
from systems.memorycpt.memorycpt import MemoryCPTLite
from systems.refind.refind import ReFind, parse_action


class ScriptedChat:
    """Fake chat model: answers by prompt type. Extraction/construction cite the turn mentioning 'Miso'."""
    model_name = "scripted"
    temperature = 0.0

    def __init__(self):
        self.agent_calls = 0

    def bind(self, **kw):
        return self

    def invoke(self, messages):
        text = messages[-1][1]
        sys = messages[0][1] if len(messages) > 1 else ""
        if "research assistant collecting evidence" in sys:
            n = sum(1 for m in messages if m[0] == "assistant")
            script = ['Thought: search.\nAction: search_chatrecord\nAction Input: {"keywords": ["cat", "name"]}',
                      'Thought: save.\nAction: take_note\nAction Input: {"indices": [1]}',
                      'Thought: done.\nAction: finish_search\nAction Input: {}']
            return AIMessage(script[min(n, 2)])
        label = next((m.group(1) for m in re.finditer(r"\[(T\d+)\] user: ([^\n]*)", text) if "Miso" in m.group(2)), None)
        if "Extract memory units" in text:
            units = [{"kind": "fact", "text": "The user's cat is named Miso.", "entities": ["Miso"], "turns": [label]}] if label else []
            units.append({"kind": "fact", "text": "Unsupported claim.", "turns": ["T999"]})
            return AIMessage(json.dumps({"units": units}))
        if "building a long-term memory store" in text:
            sem = [{"fact": "The user's cat is named Miso.", "turns": [label]}] if label else []
            return AIMessage("```json\n" + json.dumps({"episodes": [{"title": "pets", "summary": "Talked about pets.",
                                                                      "turns": ["T1"]}], "semantic": sem}) + "\n```")
        if "Write a concise summary" in text:
            return AIMessage("The user's cat is named Miso (2023-01-02). " * 200)
        return AIMessage("Miso")


class BagEmbed:
    model = "bag"

    def embed_documents(self, texts):
        out = []
        for t in texts:
            v = np.zeros(64)
            for w in re.findall(r"[a-z]+", t.lower()):
                v[hash(w) % 64] += 1
            out.append(v.tolist())
        return out


def make_q():
    turns = []
    for s in range(8):
        texts = [f"Let's discuss topic{s} and gardening tomatoes " * (3 + 4 * s), "Sure, here are tips " * 30]
        if s == 3:
            texts[0] = "I adopted a cat and named her Miso last week."
        for j, text in enumerate(texts):
            turns.append(Turn(f"s{s}:{j}", f"s{s}", datetime(2023, 1, 1 + s), ("user", "assistant")[j], text,
                              True if (s == 3 and j == 0) else None))
    return Question("q1", "single-session-user", "What is my cat's name?", datetime(2023, 2, 1), "Miso",
                    turns, ["s3:0"])


@pytest.fixture
def services(tmp_path):
    cache, log = DiskCache(tmp_path / "c.sqlite"), CostLogger(tmp_path / "costs.jsonl", Pricing({}))
    chat = LLMClient(ScriptedChat(), cache, log, method="t")
    return Services(llms={"extraction": chat, "query": chat}, embedder=EmbeddingClient(BagEmbed(), cache, log)), log


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("cls", [lambda svc: ECO(svc, use_facts=True), ReFind, MemoryCPTLite])
@pytest.mark.parametrize("budget", [40, 120, 1000])
def test_budget_provenance_and_evidence(services, cls, budget):
    svc, log = services
    q = make_q()
    m = cls(svc)
    r = m.build_context(q.question, m.build_memory(q), budget, q.question_date)
    assert r.n_tokens == count_tokens(r.context) <= budget
    assert set(r.source_turn_ids) <= {t.turn_id for t in q.haystack}
    if budget >= 120:
        assert "Miso" in r.context
        assert evidence_recall(r.source_turn_ids, q.evidence_turn_ids) == 1.0
    assert {rec.stage for rec in log.records} <= {"construction", "query_time"}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_eco_drops_units_without_provenance(services):
    svc, _ = services
    mem = ECO(svc, use_facts=True).build_memory(make_q())
    assert mem["stats"]["dropped_no_provenance"] >= 1
    assert all(f.source_turn_ids for f in mem["facts"])


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_memorycpt_summary_truncated_to_budget(services):
    svc, _ = services
    q = make_q()
    m = MemoryCPTLite(svc)
    r = m.build_context(q.question, m.build_memory(q), 200, q.question_date)
    assert r.trace["summary_truncated"] and r.n_tokens <= 200


def test_parse_action():
    assert parse_action('Thought: x\nAction: take_note\nAction Input: {"indices": [1, 2]}') == ("take_note", {"indices": [1, 2]})
    assert parse_action("no action here") == (None, {})


def test_fact_rendering_shows_resolved_event_date():
    from systems.eco.extract import MemoryUnit
    from systems.eco.method import render_fact

    u = MemoryUnit("u0", "event", "Ordered a photo album.", ["s1:2"], "2023-05-30", [], "2023-05-16")
    assert render_fact(u) == "[2023-05-16 (said 2023-05-30) | s1:2] event: Ordered a photo album."
    same = MemoryUnit("u1", "fact", "Likes tea.", ["s1:0"], "2023-05-30")
    assert render_fact(same) == "[2023-05-30 | s1:0] fact: Likes tea."
