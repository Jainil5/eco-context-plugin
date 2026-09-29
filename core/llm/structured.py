"""Helpers for LLM calls that return JSON over batches of sessions (ECO extraction, MemoryCPT-lite construction)."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass

from core.data.longmemeval import Turn
from core.llm.tokenizer import get_tokenizer


def parse_json_object(text: str) -> dict | None:
    """First top-level {...} in the reply (tolerates code fences / prose around it)."""
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                esc = (ch == "\\") and not esc
                if ch == '"' and not esc:
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


@dataclass
class Batch:
    turns: list[Turn]
    labels: dict[str, str]      # "T3" -> turn_id
    text: str                   # rendered sessions with turn labels


def session_batches(haystack: list[Turn], max_tokens: int = 6000) -> list[Batch]:
    """Group whole consecutive sessions into batches of about max_tokens (a longer session is its own batch)."""
    tok = get_tokenizer()
    sessions: list[list[Turn]] = []
    for t in haystack:
        if sessions and sessions[-1][0].session_id == t.session_id:
            sessions[-1].append(t)
        else:
            sessions.append([t])

    batches, cur, cur_tok = [], [], 0
    for s in sessions:
        n = sum(tok.count(t.text) + 8 for t in s)
        if cur and cur_tok + n > max_tokens:
            batches.append(cur)
            cur, cur_tok = [], 0
        cur.append(s)
        cur_tok += n
    if cur:
        batches.append(cur)

    out = []
    for group in batches:
        labels, lines, k = {}, [], 0
        for s in group:
            date = s[0].session_date.strftime("%Y-%m-%d (%a) %H:%M") if s[0].session_date else "unknown date"
            lines.append(f"=== Session {s[0].session_id} | date: {date} ===")
            for t in s:
                k += 1
                labels[f"T{k}"] = t.turn_id
                lines.append(f"[T{k}] {t.role}: {t.text}")
        out.append(Batch([t for s in group for t in s], labels, "\n".join(lines)))
    return out


def resolve_labels(raw, labels: dict[str, str]) -> list[str]:
    """Map the model's cited labels (["T3", 4, "t5"]) to turn ids; unknown labels are dropped."""
    if not isinstance(raw, list):
        raw = [raw]
    ids = []
    for x in raw:
        key = f"T{x}" if isinstance(x, int) else str(x).strip().upper()
        if key in labels and labels[key] not in ids:
            ids.append(labels[key])
    return ids
