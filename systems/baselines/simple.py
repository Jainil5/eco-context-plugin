"""LLM-free turn-level baselines: recent, bm25, dense, hybrid (BM25+dense RRF), mmr, oracle (upper bound)."""
from __future__ import annotations

from datetime import datetime

from core.data.longmemeval import Question
from core.context import ContextResult, fill_turns
from core.retrieval.bm25 import BM25Index


class RecentWindow:
    name = "recent_window"

    def build_memory(self, q: Question):
        return q.haystack

    def build_context(self, query, memory, token_budget, query_date: datetime | None) -> ContextResult:
        return fill_turns(list(reversed(memory)), token_budget, query_date)


class BM25Turns:
    name = "bm25_turns"

    def build_memory(self, q: Question):
        return {"turns": q.haystack, "index": BM25Index([t.text for t in q.haystack])}

    def ranked(self, query, memory):
        return [memory["turns"][i] for i, _ in memory["index"].rank(query)]

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        return fill_turns(self.ranked(query, memory), token_budget, query_date)


class OracleEvidence:
    """Gold evidence turns first, then fill with `fill_method` order. The brief specifies
    hybrid_rerank as the fill; until that exists the fill method is recorded in the name."""

    def __init__(self, fill_method=None):
        self.fill = fill_method or BM25Turns()
        self.name = f"oracle_evidence[{self.fill.name}]"

    def build_memory(self, q: Question):
        return {"gold": set(q.evidence_turn_ids), "fill": self.fill.build_memory(q)}

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        rest = self.fill.ranked(query, memory["fill"])
        gold = [t for t in rest if t.turn_id in memory["gold"]]
        others = [t for t in rest if t.turn_id not in memory["gold"]]
        return fill_turns(gold + others, token_budget, query_date)


class DenseTurns:
    name = "dense_turns"

    def __init__(self, services):
        self.services = services

    def build_memory(self, q: Question):
        from core.retrieval.dense import DenseIndex
        return {"turns": q.haystack, "qid": q.qid,
                "index": DenseIndex([t.text for t in q.haystack], self.services.embedder, qid=q.qid)}

    def ranked(self, query, memory):
        return [memory["turns"][i] for i, _ in memory["index"].rank(query, memory["qid"])]

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        return fill_turns(self.ranked(query, memory), token_budget, query_date)


class HybridRRF:
    """BM25 + dense fused with RRF (k=60). The cross-encoder rerank of the brief's `hybrid_rerank`
    is not included yet, so this is named hybrid_rrf."""
    name = "hybrid_rrf"

    def __init__(self, services):
        self.bm25, self.dense = BM25Turns(), DenseTurns(services)

    def build_memory(self, q: Question):
        return {"bm25": self.bm25.build_memory(q), "dense": self.dense.build_memory(q), "turns": q.haystack}

    def ranked(self, query, memory):
        from core.retrieval.rrf import rrf
        pos = {t.turn_id: i for i, t in enumerate(memory["turns"])}
        lists = [[pos[t.turn_id] for t in m.ranked(query, memory[k])]
                 for k, m in (("bm25", self.bm25), ("dense", self.dense))]
        return [memory["turns"][i] for i, _ in rrf(lists)]

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        return fill_turns(self.ranked(query, memory), token_budget, query_date)


class MMRTurns:
    """Maximal marginal relevance (Carbonell & Goldstein 1998) over hybrid RRF relevance and dense cosine,
    then filled in MMR order. lambda = 0.7 is a fixed standard value, not fit. The diversity control for ECO."""
    name = "mmr"

    def __init__(self, services, lam: float = 0.7):
        self.hybrid, self.lam = HybridRRF(services), lam

    def build_memory(self, q: Question):
        return self.hybrid.build_memory(q)

    def ranked(self, query, memory):
        import numpy as np
        from core.retrieval.rrf import rrf
        turns = memory["turns"]
        pos = {t.turn_id: i for i, t in enumerate(turns)}
        lists = [[pos[t.turn_id] for t in m.ranked(query, memory[k])]
                 for k, m in (("bm25", self.hybrid.bm25), ("dense", self.hybrid.dense))]
        fused = rrf(lists)
        rel = np.zeros(len(turns))
        for i, s in fused:
            rel[i] = s
        rel = rel / rel.max() if rel.max() > 0 else rel
        V = memory["dense"]["index"].vecs
        order, max_sim = [], np.zeros(len(turns))
        left = set(range(len(turns)))
        while left:
            i = max(left, key=lambda j: (self.lam * rel[j] - (1 - self.lam) * max_sim[j], -j))
            order.append(i)
            left.discard(i)
            max_sim = np.maximum(max_sim, V @ V[i])
        return [turns[i] for i in order]

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        return fill_turns(self.ranked(query, memory), token_budget, query_date)
