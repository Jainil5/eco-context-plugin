"""The one wrapper every LLM / embedding call goes through: cache + cost logging (rules 6, 7).

Wraps LangChain-style objects (the ones defined in the root `model.py`):
  - chat models: `.invoke(messages) -> AIMessage` with `.content` and `.usage_metadata`
  - embeddings:  `.embed_documents(list[str]) -> list[list[float]]`
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

from core.llm.cache import DiskCache, make_key
from core.llm.cost import CallRecord, CostLogger, UnknownPriceError
from core.llm.tokenizer import get_tokenizer

# Generation params that change outputs and therefore belong in the cache key.
_PARAM_KEYS = ("temperature", "max_tokens", "max_completion_tokens", "top_p", "seed", "stop",
               "frequency_penalty", "presence_penalty", "reasoning_effort", "num_ctx", "num_predict")


def model_id(obj: Any) -> str:
    for attr in ("model_name", "model", "model_id"):
        v = getattr(obj, attr, None)
        if isinstance(v, str) and v:
            return v
    raise ValueError(f"Cannot determine model name for {type(obj).__name__}")


def model_params(obj: Any) -> dict:
    out = {}
    for k in _PARAM_KEYS:
        v = getattr(obj, k, None)
        if v is not None:
            out[k] = v
    base_url = getattr(obj, "openai_api_base", None) or getattr(obj, "base_url", None)
    if base_url:
        out["base_url"] = str(base_url)
    return out


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    model: str


def _usd(logger: CostLogger, model: str, n_in: int, n_out: int) -> float | None:
    try:
        return logger.pricing.usd(model, n_in, n_out)
    except UnknownPriceError:
        warnings.warn(f"No price for {model!r}; cost logged as null.", stacklevel=3)
        return None


def _with_params(chat_model, params: dict):
    """Apply call-time params. ChatOllama has no `max_tokens`; its equivalent is the `num_predict` field."""
    params = dict(params)
    if "max_tokens" in params and "num_predict" in getattr(type(chat_model), "model_fields", {}):
        chat_model = chat_model.model_copy(update={"num_predict": params.pop("max_tokens")})
    return chat_model.bind(**params) if params else chat_model


def _transient(e: Exception) -> bool:
    name = type(e).__name__.lower()
    msg = str(e).lower()
    return any(s in name for s in ("timeout", "connection", "ratelimit")) or \
        any(s in msg for s in ("429", "500", "502", "503", "504", "timed out", "too many requests"))


def _invoke_with_retry(model, messages, retries: int):
    """Retry transient failures (timeouts, 429, 5xx) with exponential backoff. Latency = successful attempt."""
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            return model.invoke(messages), time.perf_counter() - t0
        except Exception as e:  # noqa: BLE001 - provider SDKs raise many types
            if attempt == retries or not _transient(e):
                raise
            time.sleep(min(60, 5 * 2 ** attempt))


class LLMClient:
    def __init__(self, chat_model: Any, cache: DiskCache, logger: CostLogger, method: str | None = None,
                 retries: int = 4):
        self.retries = retries
        self.chat_model = chat_model
        self.cache = cache
        self.logger = logger
        self.method = method
        self.model = model_id(chat_model)

    def complete(self, messages: list[dict] | str, stage: str, qid: str | None = None, **params) -> LLMResponse:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        all_params = {**model_params(self.chat_model), **params}
        key = make_key("chat", self.model, messages, all_params)

        hit = self.cache.get(key)
        if hit is not None:
            resp = LLMResponse(**{**hit, "cached": True})
        else:
            model = _with_params(self.chat_model, params)
            msg, latency = _invoke_with_retry(model, [(m["role"], m["content"]) for m in messages], self.retries)
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            usage = getattr(msg, "usage_metadata", None) or {}
            tok = get_tokenizer()
            n_in = usage.get("input_tokens") or sum(tok.count(m["content"]) for m in messages)
            n_out = usage.get("output_tokens") or tok.count(text)
            resp = LLMResponse(text=text, input_tokens=n_in, output_tokens=n_out,
                               latency_s=latency, cached=False, model=self.model)
            self.cache.set(key, {k: v for k, v in resp.__dict__.items() if k != "cached"})

        self.logger.log(CallRecord(
            stage=stage, model=self.model, input_tokens=resp.input_tokens, output_tokens=resp.output_tokens,
            latency_s=resp.latency_s, cached=resp.cached,
            usd=_usd(self.logger, self.model, resp.input_tokens, resp.output_tokens),
            method=self.method, qid=qid, kind="chat",
        ))
        return resp


class EmbeddingClient:
    """Per-text cache, so overlapping haystacks never re-embed the same turn."""

    def __init__(self, embeddings: Any, cache: DiskCache, logger: CostLogger, method: str | None = None):
        self.embeddings = embeddings
        self.cache = cache
        self.logger = logger
        self.method = method
        self.model = model_id(embeddings)

    def embed(self, texts: list[str], stage: str, qid: str | None = None, batch_size: int = 64) -> list[list[float]]:
        keys = [make_key("embed", self.model, t) for t in texts]
        out: list[list[float] | None] = [self.cache.get(k) for k in keys]
        miss = [i for i, v in enumerate(out) if v is None]

        latency = 0.0
        for s in range(0, len(miss), batch_size):
            idx = miss[s:s + batch_size]
            t0 = time.perf_counter()
            vecs = self.embeddings.embed_documents([texts[i] for i in idx])
            latency += time.perf_counter() - t0
            for i, v in zip(idx, vecs):
                out[i] = list(v)
                self.cache.set(keys[i], out[i])

        n_in = sum(get_tokenizer().count(t) for t in texts)  # approximate: our tokenizer, not the embedder's
        self.logger.log(CallRecord(
            stage=stage, model=self.model, input_tokens=n_in, output_tokens=0, latency_s=latency,
            cached=not miss, usd=_usd(self.logger, self.model, n_in, 0),
            method=self.method, qid=qid, kind="embed",
        ))
        return out  # type: ignore[return-value]
