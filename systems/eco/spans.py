"""ECO at sentence granularity: budgeted facility location over spans, with the set-level mechanisms behind flags.

Rationale. At B=250 a turn-level context holds three or four whole turns, so most of the budget goes to the filler
around the answer sentence, and the coverage objective has nothing to spread across. Sentences are the smallest unit
that still carries a claim, so the same budget buys many more distinct facts. Measured on dev (B=250, oracle view,
after the BM25 idf fix, run 20261022-112348): answer-presence 0.457 against 0.367 for turn-level ECO, so +0.090, and
turn-level ECO is *behind* BM25 on that proxy (0.431).

The selector, the retrieval stack, the cost model and the rendering do not change between the variants; only the
objective does (see systems/eco/objective.py), and systems/baselines/span.py holds granularity constant for top-k
and MMR. `eco_spans` with every mechanism off is byte-identical (282/282 context files) to the flags-off row of
20261022-112348_compare_dev_oracle_recall, which is what makes the mechanism comparison honest.

rel_tau=2 / sim_floor=0.7 are the turn-level values, re-swept for this granularity by scripts/sweep_spans.py. The
sweep is a tie: 18 cells, and across the budgets no tau or floor is consistently best (see README, "Span ECO is
tuned on its own terms"), so the pre-registered turn-level pair stays frozen.
"""
from __future__ import annotations

from core.context import fill_items, header
from core.data.longmemeval import Question
from core.llm.tokenizer import get_tokenizer
from core.retrieval.spans import build_span_index
from systems.eco.objective import build_objective
from systems.eco.select import select


class ECOspans:
    """One class, several registered names, differing only in objective flags."""

    def __init__(self, services, top_k: int = 100, rel_tau: float | None = 2.0, sim_floor: float = 0.7,
                 min_tokens: int = 8, supersede: float | None = None, one_per: str | None = None,
                 facets: int = 0, facet_weight: float = 1.0, session_weight: float = 0.0,
                 name: str = "ECO-spans"):
        self.services, self.top_k, self.min_tokens = services, top_k, min_tokens
        self.rel_tau, self.sim_floor = rel_tau, sim_floor
        self.supersede, self.one_per, self.facets, self.facet_weight = supersede, one_per, facets, facet_weight
        self.session_weight = session_weight
        self.name = name

    def build_memory(self, q: Question):
        return {"qid": q.qid, "index": build_span_index(q.haystack, self.services, q.qid, self.min_tokens)}

    def build_context(self, query, memory, token_budget, query_date=None):
        ix = memory["index"]
        fused = ix.fused(query, self.top_k)
        cand = [i for i, _ in fused]
        if not cand:
            return fill_items([], token_budget, query_date)
        obj = build_objective(ix, cand, self.rel_tau, self.sim_floor, query, supersede=self.supersede,
                             one_per=self.one_per, facets=self.facets, facet_weight=self.facet_weight,
                             session_weight=self.session_weight, fused=fused)
        avail = token_budget - get_tokenizer().count(header(query_date))  # the date line is paid for first
        S, info = select(obj.r, obj.sim, obj.fid, obj.cost, max(avail, 0), group=obj.group)
        res = fill_items([ix.spans[cand[j]].item for j in S], token_budget, query_date)
        assert len(res.trace["rank_of_chosen"]) == len(S), "selected set must fit the budget exactly"
        sessions = {ix.spans[cand[j]].turn_id.rsplit(":", 1)[0] for j in S}
        res.trace.update(info, mechanism={"supersede": self.supersede, "one_per": self.one_per,
                                          "facets": self.facets, "session_weight": self.session_weight},
                         n_spans=len(ix.spans), n_candidates=len(cand), n_selected=len(S),
                         n_turns=len({ix.spans[cand[j]].turn_index for j in S}), n_sessions_selected=len(sessions),
                         **obj.trace)
        return res
