import json

import pytest

from core.llm.cache import DiskCache, make_key
from core.llm.client import EmbeddingClient, LLMClient
from core.llm.cost import CostLogger, Pricing


def test_make_key_deterministic_and_sensitive():
    k = make_key("chat", "m", [{"role": "user", "content": "x"}], {"temperature": 0, "seed": 1})
    assert k == make_key("chat", "m", [{"role": "user", "content": "x"}], {"seed": 1, "temperature": 0})
    assert k != make_key("chat", "m2", [{"role": "user", "content": "x"}], {"temperature": 0, "seed": 1})
    assert k != make_key("chat", "m", [{"role": "user", "content": "y"}], {"temperature": 0, "seed": 1})
    assert k != make_key("chat", "m", [{"role": "user", "content": "x"}], {"temperature": 0.7, "seed": 1})


def test_cache_persists_across_instances(tmp_path):
    p = tmp_path / "c.sqlite"
    c1 = DiskCache(p); c1.set("k", {"a": [1, 2]}); c1.close()
    c2 = DiskCache(p)
    assert c2.get("k") == {"a": [1, 2]} and c2.get("missing") is None and len(c2) == 1


def test_complete_caches_and_logs(chat, cache, logger):
    client = LLMClient(chat, cache, logger, method="m")
    r1 = client.complete("what?", stage="reader", qid="q1")
    r2 = client.complete("what?", stage="reader", qid="q1")
    assert r1.text == r2.text == "answer one"
    assert chat.calls == 1, "second call must be served from cache"
    assert (r1.cached, r2.cached) == (False, True)
    # cached records keep original token counts so method cost doesn't depend on cache state
    a, b = logger.records
    assert (a.input_tokens, a.output_tokens) == (b.input_tokens, b.output_tokens)
    assert a.input_tokens > 0 and a.usd is not None
    lines = [json.loads(l) for l in logger.path.read_text().splitlines()]
    assert [l["stage"] for l in lines] == ["reader", "reader"] and lines[0]["qid"] == "q1"


def test_different_prompt_or_params_miss_cache(chat, cache, logger):
    client = LLMClient(chat, cache, logger)
    client.complete("a", stage="reader")
    client.complete("b", stage="reader")
    assert chat.calls == 2


def test_rerun_does_not_respend(chat, cache, tmp_path, pricing):
    """A fresh logger over the same cache (i.e. re-running an experiment) spends nothing."""
    LLMClient(chat, cache, CostLogger(None, pricing)).complete("q", stage="construction")
    log2 = CostLogger(None, pricing)
    LLMClient(chat, cache, log2).complete("q", stage="construction")
    s = log2.summary()
    assert chat.calls == 1 and s["actually_spent_usd"] == 0 and s["construction"]["usd"] > 0


def test_judge_excluded_from_system_cost(chat, cache, logger):
    client = LLMClient(chat, cache, logger)
    client.complete("x", stage="reader")
    client.complete("y", stage="judge")
    s = logger.summary()
    assert s["judge"]["usd"] > 0
    assert s["system_usd"] == pytest.approx(s["reader"]["usd"])


def test_unknown_stage_rejected(chat, cache, logger):
    with pytest.raises(ValueError):
        LLMClient(chat, cache, logger).complete("x", stage="misc")


def test_unknown_price_logs_null_with_warning(chat, cache):
    logger = CostLogger(None, Pricing({}))
    with pytest.warns(UserWarning):
        LLMClient(chat, cache, logger).complete("x", stage="reader")
    assert logger.records[0].usd is None


def test_embeddings_cached_per_text(embeddings, cache, logger):
    client = EmbeddingClient(embeddings, cache, logger)
    v1 = client.embed(["a", "bb"], stage="construction")
    v2 = client.embed(["bb", "ccc", "a"], stage="query_time")
    assert v1 == [[1.0, 1.0], [2.0, 1.0]]
    assert v2 == [[2.0, 1.0], [3.0, 1.0], [1.0, 1.0]]
    assert embeddings.calls == 2  # second call only embedded "ccc"
    client.embed(["a", "ccc"], stage="query_time")
    assert embeddings.calls == 2 and logger.records[-1].cached


def test_transient_errors_retried(cache, logger, monkeypatch):
    import core.llm.client as client_mod
    from langchain_core.messages import AIMessage
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)

    class Flaky:
        model_name, temperature, n = "fake-chat", 0.0, 0

        def bind(self, **kw):
            return self

        def invoke(self, msgs):
            self.n += 1
            if self.n < 3:
                raise TimeoutError("Read timed out")
            return AIMessage("ok")

    f = Flaky()
    assert LLMClient(f, cache, logger).complete("x", stage="reader").text == "ok" and f.n == 3

    class Broken(Flaky):
        def invoke(self, msgs):
            self.n += 1
            raise ValueError("bad request")

    b = Broken()
    with pytest.raises(ValueError):
        LLMClient(b, cache, logger).complete("y", stage="reader")
    assert b.n == 1  # non-transient errors are not retried
