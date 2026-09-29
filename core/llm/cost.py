"""Cost logging, pricing, and the pre-run cost guard (rules 6 and 8)."""
from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

STAGES = ("construction", "query_time", "reader", "judge")
SYSTEM_STAGES = ("construction", "query_time", "reader")  # judge is logged but not system cost


class UnknownPriceError(KeyError):
    pass


class CostLimitExceeded(RuntimeError):
    pass


@dataclass
class Price:
    input_per_1m: float
    output_per_1m: float = 0.0


class Pricing:
    """USD per 1M tokens, keyed by model name. Unknown models raise so the guard fails safe."""

    def __init__(self, table: dict[str, dict]):
        self.table = {m: Price(**p) for m, p in (table or {}).items()}

    def usd(self, model: str, input_tokens: int, output_tokens: int = 0) -> float:
        if model not in self.table:
            raise UnknownPriceError(f"No price for model {model!r}; add it under `pricing` in the config.")
        p = self.table[model]
        return (input_tokens * p.input_per_1m + output_tokens * p.output_per_1m) / 1e6


@dataclass
class CallRecord:
    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    cached: bool
    usd: float | None
    method: str | None = None
    qid: str | None = None
    kind: str = "chat"  # "chat" | "embed"
    ts: float = field(default_factory=time.time)


class CostLogger:
    """Appends one JSON line per call. Tokens/latency are the original (uncached) values, so
    reported method cost does not depend on cache state; `cached` records whether money was spent."""

    def __init__(self, path: str | Path | None = None, pricing: Pricing | None = None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.pricing = pricing or Pricing({})
        self.records: list[CallRecord] = []
        self._lock = threading.Lock()

    def log(self, rec: CallRecord) -> None:
        if rec.stage not in STAGES:
            raise ValueError(f"Unknown stage {rec.stage!r}; expected one of {STAGES}")
        with self._lock:
            self.records.append(rec)
            if self.path:
                with self.path.open("a") as f:
                    f.write(json.dumps(asdict(rec)) + "\n")

    def summary(self) -> dict:
        out: dict = {s: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "latency_s": 0.0, "usd": 0.0}
                     for s in STAGES}
        spent = 0.0
        for r in self.records:
            s = out[r.stage]
            s["calls"] += 1
            s["input_tokens"] += r.input_tokens
            s["output_tokens"] += r.output_tokens
            s["latency_s"] += r.latency_s
            s["usd"] += r.usd or 0.0
            if not r.cached:
                spent += r.usd or 0.0
        out["system_usd"] = sum(out[s]["usd"] for s in SYSTEM_STAGES)
        out["actually_spent_usd"] = spent
        return out


@dataclass
class PlannedCalls:
    """One line of a cost estimate: n calls of a model with average token counts."""
    model: str
    n_calls: int
    avg_input_tokens: int
    avg_output_tokens: int
    stage: str = ""


def estimate_usd(plan: list[PlannedCalls], pricing: Pricing) -> float:
    return sum(pricing.usd(p.model, p.n_calls * p.avg_input_tokens, p.n_calls * p.avg_output_tokens)
               for p in plan)


def cost_guard(plan: list[PlannedCalls], pricing: Pricing, max_usd: float | None, printer=print) -> float:
    """Print the estimate; raise CostLimitExceeded if above max_usd. Does not discount cache hits,
    so the estimate is an upper bound. max_usd=None disables the limit (estimate is still printed,
    models without a price are shown as unpriced instead of refusing)."""
    total = 0.0
    printer("Cost estimate (upper bound, ignores cache hits):")
    for p in plan:
        try:
            usd = pricing.usd(p.model, p.n_calls * p.avg_input_tokens, p.n_calls * p.avg_output_tokens)
        except UnknownPriceError:
            if max_usd is not None:
                raise
            usd = None
        total += usd or 0.0
        printer(f"  {p.stage or '-':<13} {p.model:<28} {p.n_calls:>7} calls  "
                f"~{p.avg_input_tokens}/{p.avg_output_tokens} tok in/out  "
                + (f"${usd:.4f}" if usd is not None else "unpriced"))
    if max_usd is None:
        printer(f"  TOTAL ${total:.4f}  (no max_usd limit set)")
        return total
    printer(f"  TOTAL ${total:.4f}  (max_usd = ${max_usd:.2f})")
    if total > max_usd:
        raise CostLimitExceeded(f"Estimated ${total:.4f} exceeds max_usd=${max_usd:.2f}. Ask before running.")
    return total
