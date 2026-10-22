"""Evidence metrics.

`evidence_recall` / `verbatim_recall` are turn-level: a gold turn counts once its id is cited / its raw text is
in the context. Both are blind to granularity — a context holding one answer sentence from a 400-token turn and
one holding the whole turn score differently only through the id, never by how much gold content arrived.

`content_recall` fixes that by scoring the share of a gold turn's *sentences* whose raw text reached the reader,
so turn-level and span-level systems are comparable on one scale. A whole-turn rendering scores exactly as it
does turn-level (every sentence of an included turn is present as a substring), so this is a refinement, not a
replacement.
"""
from __future__ import annotations

import re

from core.text import normalize_ws, sentences

_MIN_CHARS = 40          # shorter fragments ("Yes.", "Thanks!") match too easily to be worth scored
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = frozenset("""a about after all an and any are as at be because before being but by can did do does for
from had has have he her his how i in into is it its just like me more my of on or our out she should so some
than that the their them then there these they this those to too was we were what when where which who will
with you your""".split())


def evidence_recall(source_turn_ids, evidence_turn_ids) -> float | None:
    """Fraction of gold evidence turns included in the context; None if the question has no evidence."""
    if not evidence_turn_ids:
        return None
    s = set(source_turn_ids)
    return sum(e in s for e in evidence_turn_ids) / len(evidence_turn_ids)


def content_recall(context: str, gold_texts: list[str]) -> float | None:
    """Share of the gold turns' sentences whose exact raw text appears in the context; None if no gold text.

    Granularity-neutral: a turn-level system that includes a whole turn scores every sentence of it, and a
    span-level system gets partial credit for the sentences it actually delivered.
    """
    qs = [q for g in gold_texts for q in sentences(g, _MIN_CHARS)]
    if not qs:
        return None
    hay = normalize_ws(context).lower()
    return sum(normalize_ws(q).lower() in hay for q in qs) / len(qs)


def answer_in_context(context: str, answer) -> float | None:
    """Did the gold answer itself reach the reader? 1.0 exact phrase, 0.5 all content words, 0.0 absent.

    Turn and sentence recall both miss what a reader actually needs: a context that swaps a gold turn for the
    assistant's paraphrase of it can still contain the answer. Abstention questions have no answer string to
    find, so they return None and are excluded from the metric.
    """
    if answer is None:
        return None
    s = str(answer).strip().lower()
    if not s or s.startswith("you did not mention"):
        return None
    hay = normalize_ws(context).lower()
    if s in hay:
        return 1.0
    toks = {t for t in _TOKEN.findall(s) if t not in _STOP}
    if toks and all(t in hay for t in toks):
        return 0.5
    return 0.0
