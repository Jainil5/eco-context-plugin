"""BM25 over short texts (rank_bm25 Okapi). Optional Porter stemming + stopword removal (ReFind setting)."""
from __future__ import annotations

import re
import threading
import warnings

from rank_bm25 import BM25Okapi

try:  # Porter stemming is only used by ReFind (paper setting); optional so a missing package can't block a run
    import snowballstemmer
except ImportError:  # pragma: no cover
    snowballstemmer = None

# snowballstemmer stemmers keep mutable state and are not thread-safe: one instance per thread.
_local = threading.local()


def _stemmer():
    if not hasattr(_local, "stem"):
        _local.stem = snowballstemmer.stemmer("porter")
    return _local.stem

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("""a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing down during each few for from further had has have having he her
here hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off
on once only or other our ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was we were what when where which
while who whom why will with you your yours yourself yourselves""".split())


def tokenize(text: str, stem: bool = False) -> list[str]:
    toks = _TOKEN.findall(text.lower())
    if stem:
        toks = [t for t in toks if t not in STOPWORDS]
        if snowballstemmer is None:
            warnings.warn("snowballstemmer not installed: BM25 runs without Porter stemming", stacklevel=2)
        else:
            toks = _stemmer().stemWords(toks)
    return toks


class BM25Index:
    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75, stem: bool = False):
        self.stem = stem
        self.bm25 = BM25Okapi([tokenize(t, stem) or ["<empty>"] for t in texts], k1=k1, b=b)

    def scores(self, query: str):
        return self.bm25.get_scores(tokenize(query, self.stem))

    def rank(self, query: str) -> list[tuple[int, float]]:
        return sorted(enumerate(self.scores(query)), key=lambda x: (-x[1], x[0]))
