"""Deterministic derived facts: the second rendering kind ECO can select over.

Every other system in this repo renders context either as raw text (turns, spans) or as LLM-generated summaries
(MemoryCPT, eco_facts). This module adds a third kind that costs no model calls at all: a fact *derived* from the
pool by pure symbolic aggregation - how many times a query facet was mentioned, across how many conversations,
first and most recent date. Such an item is ~20 tokens where its evidence is 20x that, and it states a relation
(a count, a recency) that no single span contains - which is exactly the shape of the temporal-reasoning and
counting questions where verbatim retrieval has a ceiling the reader cannot cross by reading more sentences.

The items are honest about being derived ("[derived]" header, source turn ids kept, verbatim=False) and only
generated for facets whose mentions span at least two dated sessions - a fact about one mention is that mention's
text, not a fact.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from core.context import Item
from core.llm.tokenizer import get_tokenizer
from systems.eco.objective import facet_terms

DERIVE_MIN_SPANS = 2      # a derived fact must aggregate at least two mentions
DERIVE_MIN_SESSIONS = 2   # ... across at least two conversations, else it states nothing no span states

COUNT_CUE = re.compile(r"\bhow (?:many|much)\b|\bnumber of\b|\bhow often\b", re.I)

COUNT_CAP_TOKENS = 60   # a derived fact must stay cheap to render, whatever the data


@dataclass
class Derived:
    item: Item
    covered: list[int]     # indices into the candidate list these spans were aggregated from
    n_mentions: int
    n_sessions: int
    first_date: datetime
    last_date: datetime


def _fmt(d: datetime | None) -> str:
    return d.strftime("%Y-%m-%d") if d else "unknown date"


def derive_facts(index, cand: list[int], query: str, query_date: datetime | None,
                 max_facts: int = 4) -> list[Derived]:
    """Facet-aggregation facts over the ranked candidate spans (cand indexes into index.spans)."""
    out: list[Derived] = []
    counting = bool(COUNT_CUE.search(query))
    n_terms = 4 if counting else 3
    for term in facet_terms(query, index.bm25, cand, n_terms):
        sc = index.bm25.scores(term)
        hits = [j for j, i in enumerate(cand) if sc[i] > 0]
        if len(hits) < DERIVE_MIN_SPANS:
            continue
        sessions = {index.spans[cand[j]].turn_id.rsplit(":", 1)[0] for j in hits}
        dates = [index.turns[index.spans[cand[j]].turn_index].session_date for j in hits
                 if index.spans[cand[j]].turn_index < len(index.turns)]
        dated = [d for d in dates if d is not None]
        if len(sessions) < DERIVE_MIN_SESSIONS and not counting:
            continue
        first, last = (min(dated), max(dated)) if dated else (None, None)
        src = list(dict.fromkeys(index.spans[cand[j]].turn_id for j in hits))
        if counting:
            text = f"[derived] '{term}' is mentioned {len(hits)} times." + _dates_clause(dated, term, len(hits))
        else:
            text = (f"[derived] '{term}': {len(hits)} mentions across {len(sessions)} conversations, "
                    f"first {_fmt(first)}, most recent {_fmt(last)}.")
        key = (last or first or datetime.min, "zz-derive", len(out), 0)
        out.append(Derived(Item(text, src, key, verbatim=False), hits, len(hits), len(sessions), first, last))
        if len(out) >= max_facts:
            break
    return list({d.item.text: d for d in out}.values())


def _dates_clause(dated: list[datetime], term: str, n_hits: int, cap: int = COUNT_CAP_TOKENS) -> str:
    """List the most recent dates, dropping the oldest until the full fact fits `cap` tokens.

    The count in the sentence stays the true count, so a truncated list reads "+N more" rather than
    understating the mentions - and no fact can ever exceed the render cap, whatever the data does.
    """
    if not dated:
        return ""
    head = "; dates: "
    stem = f"[derived] '{term}' is mentioned {n_hits} times."
    for k in range(min(8, len(dated)), 0, -1):
        shown = sorted(dated)[-k:]
        more = len(dated) - k
        tail = f" (+{more} more)" if more else ""
        clause = f"{head}{', '.join(_fmt(d) for d in shown)}{tail}"
        if get_tokenizer().count(stem + clause) + 1 <= cap:
            return clause
    return ""
