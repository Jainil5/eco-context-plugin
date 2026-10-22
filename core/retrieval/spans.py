"""Span-level candidate index: sentence-sized renderings of the haystack, retrieved by BM25 + dense.

The comparison this repo makes is about *what a budget buys*. Turn and span granularity are two answers to that
question, so they need the same retrieval stack and the same selectors on top: this module builds the shared
index, and systems/baselines/span.py plus systems/eco/spans.py put different selectors on it.
"""
from __future__ import annotations

from dataclasses import dataclass

from core.context import Span, span_units
from core.llm.tokenizer import get_tokenizer
from core.retrieval.bm25 import BM25Index
from core.retrieval.dense import DenseIndex
from core.retrieval.rrf import rrf


@dataclass
class SpanIndex:
    qid: str
    turns: list
    spans: list[Span]
    bm25: BM25Index
    dense: DenseIndex

    def fused(self, query: str, top_k: int) -> list[tuple[int, float]]:
        """RRF over BM25 and dense span rankings, best first, truncated to top_k spans."""
        b = [i for i, _ in self.bm25.rank(query)]
        d = [i for i, _ in self.dense.rank(query, self.qid)]
        return rrf([b, d])[:top_k]


def build_span_index(turns, services, qid: str, min_tokens: int = 8) -> SpanIndex:
    """One BM25 + dense key per span, over the span's raw sentence text (no date/role header)."""
    spans = span_units(turns, get_tokenizer(), min_tokens=min_tokens)
    texts = [s.text for s in spans] or [""]
    return SpanIndex(qid=qid, turns=turns, spans=spans, bm25=BM25Index(texts),
                     dense=DenseIndex(texts, services.embedder, qid=qid))
