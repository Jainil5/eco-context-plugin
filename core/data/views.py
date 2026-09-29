"""Data views: which part of a LongMemEval haystack a system gets to choose from.

oracle: only the evidence sessions (`answer_session_ids`), as in LongMemEval's official oracle file. Tests
        context optimization under a budget without the search problem of a ~115K-token haystack.
full:   the whole haystack.
"""
from __future__ import annotations

from dataclasses import replace

from core.data.longmemeval import Question


def oracle_view(q: Question) -> Question:
    keep = set(q.evidence_session_ids)
    return replace(q, haystack=[t for t in q.haystack if t.session_id.split("#")[0] in keep])


def apply_view(q: Question, view: str) -> Question:
    if view == "full":
        return q
    if view == "oracle":
        return oracle_view(q)
    raise ValueError(f"unknown view {view!r} (oracle | full)")
