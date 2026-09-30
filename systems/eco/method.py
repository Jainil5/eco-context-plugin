"""ECO: budgeted facility-location selection over raw turns, optionally with LLM-extracted facts.

use_facts=False ("eco"): optimizer only. Units are raw turns, relevance = RRF(BM25, dense), no LLM calls.
use_facts=True ("eco_facts", the POC v2 setup):
- Units are raw turns. LLM-extracted facts are (a) extra retrieval keys for the turns they cite (key expansion,
  LongMemEval paper) and (b) an alternative, cheaper "fact" rendering of their primary (longest) cited turn.
  Selection keeps at most one rendering per unit (brief §5) and may upgrade fact -> raw when budget allows.
- Candidates = BM25 top-k U dense top-k over all keys, mapped to units; r = normalized RRF score (uniform
  weights, nothing fit on dev yet); sim = cosine between the units' raw-turn embeddings.
- fid: raw 1.0; fact 0.8 x retention, retention = min(1, max cos(query, fact) / cos(query, raw turn)), so a
  fact that dropped what the query asks about loses fidelity. 0.8 is fixed, not fit.
- Facts are shown with the LLM-resolved event date; context is ordered by event date.
Not yet: query intent, closure constraints, supersession policy, graph/PPR, cross-encoder, learned weights.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np

from core.data.longmemeval import Question
from systems.eco.extract import MemoryUnit, extract_units
from systems.eco.select import select
from core.llm.tokenizer import get_tokenizer
from core.context import ContextResult, Item, fill_items, header, render_turn, turn_sort_key
from core.retrieval.bm25 import BM25Index
from core.retrieval.dense import DenseIndex
from core.retrieval.rrf import rrf

FID_RAW = 1.0
FID_FACT = 0.8


def _day(s: str) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y-%m-%d")
    except ValueError:
        return None


def render_fact(u: MemoryUnit) -> str:
    when = u.event_date if u.event_date and u.event_date != u.date else u.date
    said = f" (said {u.date})" if when != u.date else ""
    return f"[{when}{said} | {u.source_turn_ids[0]}] {u.kind}: {u.text}"


def _first_ranks(ranking: list[int], key_unit: list[int], top_k: int) -> list[int]:
    """Key ranking -> unit ranking (a unit is placed at its best key), truncated to top_k units."""
    out, seen = [], set()
    for k in ranking:
        u = key_unit[k]
        if u not in seen:
            seen.add(u)
            out.append(u)
            if len(out) == top_k:
                break
    return out


class ECO:
    def __init__(self, services, use_facts: bool = False, top_k: int = 100, construction_parallel: int = 4,
                 rel_tau: float | None = 2.0, sim_floor: float = 0.7):
        """rel_tau: None = relevance is the normalized RRF score (nearly flat over ranks); a number = rank-decayed
        relevance exp(-(rank - 1) / rel_tau), which focuses coverage on the question.
        sim_floor: cosine similarity at or below this counts as no coverage; above it is rescaled to (0, 1].
        Defaults chosen on the DEV split only (scripts/sweep_eco.py, run 20260930-073910_eco_sweep_dev, rule: best
        mean verbatim recall over B in {250, 500, 1000, 2000}); before tuning they were None / 0.0."""
        self.services, self.use_facts, self.top_k, self.parallel = services, use_facts, top_k, construction_parallel
        self.rel_tau, self.sim_floor = rel_tau, sim_floor
        self.name = "ECO+facts" if use_facts else "ECO"

    def build_memory(self, q: Question):
        if self.use_facts:
            facts, stats = extract_units(q, self.services.llm("extraction"), parallel=self.parallel)
        else:
            facts, stats = [], {}
        turns = q.haystack
        pos = {t.turn_id: i for i, t in enumerate(turns)}
        tok = get_tokenizer()

        # Keys: every raw turn, plus every fact for each turn it cites.
        key_text = [t.text for t in turns]
        key_unit = list(range(len(turns)))
        key_fact = [-1] * len(turns)  # index into facts, -1 for raw keys
        attached: dict[int, list[int]] = {}  # unit -> facts rendered with it
        for f_i, f in enumerate(facts):
            cited = [pos[i] for i in f.source_turn_ids if i in pos]
            if not cited:
                continue
            for u in dict.fromkeys(cited):
                key_text.append(f.text)
                key_unit.append(u)
                key_fact.append(f_i)
            primary = max(cited, key=lambda u: (len(turns[u].text), -u))
            attached.setdefault(primary, []).append(f_i)

        # Renderings: raw for every unit, fact for units with attached facts.
        rend = []  # dicts: unit, kind, item, cost, facts
        for u, t in enumerate(turns):
            it = Item(render_turn(t), [t.turn_id], turn_sort_key(t))
            rend.append({"unit": u, "kind": "raw", "item": it, "cost": tok.count(it.text) + 1, "facts": []})
            if u in attached:
                fs = [facts[i] for i in attached[u]]
                text = "\n".join(render_fact(f) for f in fs)
                ids = list(dict.fromkeys(i for f in fs for i in f.source_turn_ids))
                when = min((_day(f.event_date) for f in fs if _day(f.event_date)), default=None)
                key = (when or t.session_date or datetime.min,) + turn_sort_key(t)[1:]
                rend.append({"unit": u, "kind": "fact", "item": Item(text, ids, key, verbatim=False),
                             "cost": tok.count(text) + 1, "facts": attached[u]})
        by_unit: dict[int, list[int]] = {}
        for j, rd in enumerate(rend):
            by_unit.setdefault(rd["unit"], []).append(j)

        dense = DenseIndex(key_text, self.services.embedder, qid=q.qid)
        return {"qid": q.qid, "turns": turns, "facts": facts, "rend": rend, "by_unit": by_unit,
                "key_unit": key_unit, "key_fact": key_fact, "bm25": BM25Index(key_text), "dense": dense,
                "stats": stats}

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        qid, rend, key_unit = memory["qid"], memory["rend"], memory["key_unit"]
        n_units = len(memory["turns"])
        dense: DenseIndex = memory["dense"]
        qv = dense.query_vec(query, qid)
        key_cos = dense.vecs @ qv
        b = _first_ranks([i for i, _ in memory["bm25"].rank(query)], key_unit, self.top_k)
        d = _first_ranks(sorted(range(len(key_cos)), key=lambda i: (-key_cos[i], i)), key_unit, self.top_k)
        fused = rrf([b, d])
        units = [u for u, _ in fused]
        r_unit = np.array([s for _, s in fused])
        if self.rel_tau:
            r_unit = np.exp(-np.arange(len(units)) / self.rel_tau)  # fused order is best first
        else:
            r_unit = r_unit / r_unit.max()

        C, group, r, fid = [], [], [], []
        fact_cos = {f: c for f, c in zip(memory["key_fact"], key_cos) if f >= 0}
        for g, u in enumerate(units):
            raw_cos = max(float(key_cos[u]), 1e-6)  # raw keys are indexed by unit id
            for j in memory["by_unit"][u]:
                rd = rend[j]
                if rd["kind"] == "raw":
                    f = FID_RAW
                else:
                    f = FID_FACT * min(1.0, max(float(fact_cos[i]) for i in rd["facts"]) / raw_cos)
                C.append(j), group.append(g), r.append(r_unit[g]), fid.append(max(f, 0.0))
        r, fid, group = np.array(r), np.array(fid), np.array(group)
        V = dense.vecs[[rend[j]["unit"] for j in C]]  # unit identity = its raw turn's embedding
        sim = np.clip((V @ V.T - self.sim_floor) / (1.0 - self.sim_floor), 0.0, None)
        np.fill_diagonal(sim, 1.0)
        sim[group[:, None] == group[None, :]] = 1.0  # renderings of the same unit
        cost = np.array([rend[j]["cost"] for j in C], dtype=float)

        avail = token_budget - get_tokenizer().count(header(query_date))
        S, info = select(r, sim, fid, cost, max(avail, 0), group=group)
        chosen = [rend[C[j]]["item"] for j in S]
        res = fill_items(chosen, token_budget, query_date)
        assert len(res.trace["rank_of_chosen"]) == len(chosen), "selected set must fit the budget exactly"
        kinds = [rend[C[j]]["kind"] for j in S]
        res.trace.update(info, n_units=n_units, n_candidates=len(C), n_selected=len(S),
                         n_raw=kinds.count("raw"), n_fact=kinds.count("fact"), construction=memory["stats"])
        return res
