"""Shared ContextMethod interface (brief §4) and the budgeted fill + render used by turn-level baselines."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from core.data.longmemeval import Question, Turn
from core.llm.tokenizer import get_tokenizer


@dataclass
class ContextResult:
    context: str                     # exactly what the reader sees (<= token_budget)
    n_tokens: int
    source_turn_ids: list[str]       # every turn the context draws on (raw or via a fact/summary)
    trace: dict = field(default_factory=dict)
    verbatim_turn_ids: list[str] = field(default_factory=list)  # turns whose raw text is in the context


class ContextMethod(Protocol):
    name: str
    def build_memory(self, q: Question) -> Any: ...
    def build_context(self, query: str, memory: Any, token_budget: int,
                      query_date: datetime | None) -> ContextResult: ...


def render_turn(t: Turn) -> str:
    date = t.session_date.strftime("%Y-%m-%d") if t.session_date else "unknown date"
    return f"[{date} | {t.turn_id}] {t.role}: {t.text}"


def header(query_date: datetime | None) -> str:
    return f"Current date: {query_date.strftime('%Y-%m-%d')}\n" if query_date else ""


@dataclass
class Item:
    """One renderable piece of context with its provenance."""
    text: str
    source_turn_ids: list[str]
    sort_key: tuple = ()
    verbatim: bool = True            # False for derived renderings (facts, summaries)


def turn_sort_key(t: Turn) -> tuple:
    sid, idx = t.turn_id.rsplit(":", 1)
    return (t.session_date or datetime.min, sid, int(idx))


def turn_item(t: Turn) -> Item:
    return Item(render_turn(t), [t.turn_id], turn_sort_key(t))


def fill_items(ranked: list[Item], budget: int, query_date: datetime | None,
               chronological: bool = True) -> ContextResult:
    """Add whole items in rank order while they fit; an item that does not fit is skipped (logged) and
    filling continues with the next. Selected items are re-sorted chronologically for the reader."""
    tok = get_tokenizer()
    head = header(query_date)
    if tok.count(head) > budget:  # tiny budgets: drop the date header rather than exceed B
        head = ""
    used = tok.count(head)
    chosen: list[tuple[int, Item]] = []
    skipped = 0
    for rank, it in enumerate(ranked):
        c = tok.count(it.text) + 1  # +1 for the newline separator
        if used + c <= budget:
            chosen.append((rank, it))
            used += c
        else:
            skipped += 1
    if chronological:
        chosen.sort(key=lambda x: x[1].sort_key)
    context = head + "\n".join(it.text for _, it in chosen)
    n = tok.count(context)
    assert n <= budget, f"context {n} > budget {budget}"
    ids = list(dict.fromkeys(i for _, it in chosen for i in it.source_turn_ids))
    verbatim = list(dict.fromkeys(i for _, it in chosen if it.verbatim for i in it.source_turn_ids))
    return ContextResult(context=context, n_tokens=n, source_turn_ids=ids, verbatim_turn_ids=verbatim,
                         trace={"rank_of_chosen": [r for r, _ in chosen], "skipped_too_long": skipped,
                                "n_candidates": len(ranked)})


def fill_turns(ranked: list[Turn], budget: int, query_date: datetime | None,
               chronological: bool = True) -> ContextResult:
    return fill_items([turn_item(t) for t in ranked], budget, query_date, chronological)


@dataclass
class Services:
    """LLM / embedding clients handed to methods. Roles: extraction, query (query-time agent/summarizer)."""
    llms: dict[str, Any] = field(default_factory=dict)
    embedder: Any = None

    def llm(self, role: str):
        if role not in self.llms:
            raise KeyError(f"No LLM client for role {role!r}")
        return self.llms[role]
