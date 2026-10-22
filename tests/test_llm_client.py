import json

import pytest
from langchain_core.messages import AIMessage

from core.llm.cache import DiskCache, make_key
from core.llm.client import EmbeddingClient, EmptyResponseError, LLMClient, ProviderError, model_params
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


class ScriptedReply:
    """Minimal chat stub whose reply is exactly the AIMessage we want to test the unwrapping against."""
    model_name, temperature = "fake-chat", 0.0

    def __init__(self, msgs):
        self.msgs, self.calls, self.bound = list(msgs), 0, []

    def bind(self, **kw):
        self.bound.append(kw)
        return self

    def invoke(self, msgs):
        self.calls += 1
        return self.msgs[min(self.calls, len(self.msgs)) - 1]


# what the NVIDIA NIM route actually returns for an openai/gpt-oss-20b request with default reasoning effort
NIM_ERROR_MSG = AIMessage(content="", response_metadata={
    "error": {"message": "unexpected tokens remaining in message header: Some(\"to=assistant\")",
              "type": "BadRequestError", "code": 400},
    "finish_reason": "stop"})


def test_provider_error_hidden_in_metadata_is_fatal(cache, logger, no_sleep):
    """200 + empty content + an error dict in the metadata used to be cached as a successful empty reply, which is
    how a whole system ended up scoring 0. It must raise, retry nothing, and cache nothing."""
    m = ScriptedReply([NIM_ERROR_MSG])
    with pytest.raises(ProviderError, match="unexpected tokens remaining"):
        LLMClient(m, cache, logger).complete("x", stage="reader")
    assert m.calls == 1 and no_sleep == [] and len(cache) == 0


def test_empty_completion_is_retried_then_raised(cache, logger, no_sleep):
    m = ScriptedReply([AIMessage("")] * 3)
    with pytest.raises(EmptyResponseError):
        LLMClient(m, cache, logger, retries=2).complete("x", stage="reader")
    assert m.calls == 3 and len(no_sleep) == 2 and len(cache) == 0


def test_empty_completion_that_recovers_is_cached(cache, logger, no_sleep):
    m = ScriptedReply([AIMessage(""), AIMessage("recovered")])
    assert LLMClient(m, cache, logger).complete("x", stage="reader").text == "recovered"
    assert m.calls == 2 and len(cache) == 1


def test_defaults_reach_the_call_and_the_cache_key(chat, cache, logger):
    """`defaults` are what make hosted reasoning models usable; they must apply to every call and must not let a
    differently-parametered entry be served from cache."""
    m = ScriptedReply([AIMessage("ok")])
    LLMClient(m, cache, logger, defaults={"reasoning_effort": "low"}).complete("x", stage="reader", max_tokens=4096)
    assert m.bound == [{"reasoning_effort": "low", "max_tokens": 4096}], "call params win over defaults"
    plain = ScriptedReply([AIMessage("ok2")])
    LLMClient(plain, cache, logger).complete("x", stage="reader", max_tokens=4096)
    assert plain.calls == 1, "a different param set is a different cache entry"
    again = ScriptedReply([AIMessage("ok3")])
    assert LLMClient(again, cache, logger, defaults={"reasoning_effort": "low"}).complete(
        "x", stage="reader", max_tokens=4096).cached
    assert again.calls == 0


def test_cached_blank_reply_is_refreshed(chat, cache, logger):
    """Poisoned entries written before this check existed must heal on the next run, not be served forever."""
    msgs = [{"role": "user", "content": "x"}]
    key = make_key("chat", "fake-chat", msgs, model_params(chat))
    cache.set(key, {"text": "  ", "input_tokens": 5, "output_tokens": 0, "latency_s": 0.1, "model": "fake-chat"})
    r = LLMClient(chat, cache, logger).complete("x", stage="reader")
    assert r.text == "answer one" and chat.calls == 1 and not r.cached
    assert cache.get(key)["text"] == "answer one"
