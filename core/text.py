"""Text utilities shared by the systems and the metrics, so a "sentence" means the same thing in both."""
from __future__ import annotations

import re

_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_STRIP = re.compile(r"\s+")


def sentences(text: str, min_chars: int = 40) -> list[str]:
    """Split raw turn text into sentence-sized pieces, keeping original casing and punctuation.

    Pieces shorter than `min_chars` are merged into the previous one: a 3-token fragment is not worth a context
    line, and it matches too easily to be a useful scoring unit. Returned pieces are contiguous substrings of
    `text`, which is what lets a metric check verbatim inclusion by plain substring test.
    """
    out: list[str] = []
    for piece in _SPLIT.split(text.strip()):
        p = piece.strip()
        if not p:
            continue
        if out and len(p) < min_chars:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def normalize_ws(text: str) -> str:
    return _STRIP.sub(" ", text).strip()
