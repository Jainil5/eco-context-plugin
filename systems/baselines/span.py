"""Span-level baselines: the same selectors as systems/baselines/simple.py, applied to sentence candidates.

These exist to make the granularity comparison honest. If only ECO moved to spans, a win could be credited to the
sentences rather than to the coverage objective; running top-k and MMR over the identical span pool separates the
two effects. Both use the same fused BM25 + dense relevance the turn-level systems use.
"""
from __future__ import annotations

import numpy as np

from core.context import fill_items
from core.data.longmemeval import Question
from core.retrieval.mmr import mmr_order
from core.retrieval.spans import build_span_index


class SpanHybridRRF:
    """Fill the budget with the highest-scoring spans, in fused BM25 + dense order."""
    name = "span_hybrid_rrf"

    def __init__(self, services):
        self.services = services

    def build_memory(self, q: Question):
        return {"qid": q.qid, "index": build_span_index(q.haystack, self.services, q.qid)}

    def ranked(self, query, memory):
        ix = memory["index"]
        return [ix.spans[i] for i, _ in ix.fused(query, len(ix.spans))]

    def build_context(self, query, memory, token_budget, query_date=None):
        return fill_items([s.item for s in self.ranked(query, memory)], token_budget, query_date)


class SpanMMR:
    """MMR (lambda 0.7, the turn-level setting) over the same span pool."""
    name = "span_mmr"

    def __init__(self, services, lam: float = 0.7):
        self.services, self.lam = services, lam

    def build_memory(self, q: Question):
        return {"qid": q.qid, "index": build_span_index(q.haystack, self.services, q.qid)}

    def ranked(self, query, memory):
        ix = memory["index"]
        rel = np.zeros(len(ix.spans))
        for i, s in ix.fused(query, len(ix.spans)):
            rel[i] = s
        if rel.max() > 0:
            rel = rel / rel.max()
        return [ix.spans[i] for i in mmr_order(rel, ix.dense.vecs, self.lam)]

    def build_context(self, query, memory, token_budget, query_date=None):
        return fill_items([s.item for s in self.ranked(query, memory)], token_budget, query_date)
