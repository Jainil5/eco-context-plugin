"""Dataset statistics printed after loading (brief §3)."""
from __future__ import annotations

from collections import Counter

import numpy as np

from core.data.longmemeval import Question
from core.llm.tokenizer import get_tokenizer


def turn_text(t) -> str:
    return f"{t.role}: {t.text}"


def haystack_tokens(q: Question) -> int:
    enc = get_tokenizer()._enc
    return sum(len(x) for x in enc.encode_ordinary_batch([turn_text(t) for t in q.haystack], num_threads=8))


def _dist(xs) -> str:
    a = np.asarray(xs)
    p = np.percentile(a, [0, 25, 50, 75, 100])
    return (f"mean {a.mean():,.0f} | min {p[0]:,.0f} p25 {p[1]:,.0f} median {p[2]:,.0f} "
            f"p75 {p[3]:,.0f} max {p[4]:,.0f}")


def dataset_stats(questions: list[Question], name: str = "") -> str:
    lines = [f"== {name} ({len(questions)} questions) =="]
    lines.append("questions per type (abstention in brackets):")
    types = Counter(q.qtype for q in questions)
    abst = Counter(q.qtype for q in questions if q.is_abstention)
    for t, n in sorted(types.items()):
        lines.append(f"  {t:<28} {n:>4}  [{abst.get(t, 0)} abs]")
    toks = [haystack_tokens(q) for q in questions]
    lines.append(f"haystack tokens (o200k): {_dist(toks)}")
    lines.append(f"sessions per haystack:   {_dist([len({t.session_id for t in q.haystack}) for q in questions])}")
    lines.append(f"turns per haystack:      {_dist([len(q.haystack) for q in questions])}")
    ev = [len(q.evidence_turn_ids) for q in questions]
    lines.append(f"evidence turns per q:    {_dist(ev)}")
    no_ev = Counter(q.stratum for q in questions if not q.evidence_turn_ids)
    lines.append(f"questions with 0 evidence turns: {sum(no_ev.values())} {dict(no_ev)}")
    return "\n".join(lines)
