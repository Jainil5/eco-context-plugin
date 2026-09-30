"""ECO construction (POC): one extraction call per session batch -> provenance-linked memory units.
Raw turns are always kept as units too, so extraction errors are recoverable."""
from __future__ import annotations

import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core.data.longmemeval import Question
from core.llm.structured import parse_json_object, resolve_labels, session_batches

_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
KINDS = ("fact", "event", "decision", "preference", "goal")

EXTRACT_PROMPT = """Extract memory units from the chat sessions below between a user and an assistant.
Output JSON: {{"units": [{{"kind": one of {kinds}, "text": one self-contained sentence with concrete details \
(names, numbers, places) and absolute dates, "entities": [key entities], "event_date": "YYYY-MM-DD" when the event happened or the fact \
was stated (resolve relative dates against the session date; use the session date if unknown), "turns": [the [T#] \
labels that support it]}}]}}
Rules: every unit must cite the turns it comes from; resolve relative dates ("yesterday", "two weeks ago") \
against the session date; include facts the assistant stated if they are specific (recommendations, lists, \
numbers); skip greetings and generic chit-chat. Output only the JSON object.

{sessions}"""


@dataclass
class MemoryUnit:
    uid: str
    kind: str                       # "raw_turn" | fact | event | decision | preference | goal
    text: str
    source_turn_ids: list[str]
    date: str                       # YYYY-MM-DD of the first source turn's session
    entities: list[str] = field(default_factory=list)
    event_date: str = ""            # YYYY-MM-DD the event happened (LLM-resolved); "" = same as date


def extract_units(q: Question, llm, batch_tokens: int = 6000, parallel: int = 8) -> tuple[list[MemoryUnit], dict]:
    date_of = {t.turn_id: (t.session_date.strftime("%Y-%m-%d") if t.session_date else "unknown") for t in q.haystack}
    units: list[MemoryUnit] = []
    stats = {"batches": 0, "unparsed_batches": 0, "dropped_no_provenance": 0}
    batches = session_batches(q.haystack, batch_tokens)
    def call(b):
        try:
            return llm.complete(EXTRACT_PROMPT.format(kinds="|".join(KINDS), sessions=b.text),
                                stage="construction", qid=q.qid, max_tokens=16384)
        except Exception:  # noqa: BLE001 - a batch that fails after retries is counted, not fatal
            return None

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        replies = list(ex.map(call, batches))  # batches are independent; map keeps order
    for b, reply in zip(batches, replies):
        stats["batches"] += 1
        if reply is None:
            stats["failed_batches"] = stats.get("failed_batches", 0) + 1
            continue
        if reply.output_tokens >= 16384:  # hit the output cap: JSON is likely cut off
            stats["truncated_batches"] = stats.get("truncated_batches", 0) + 1
        data = parse_json_object(reply.text)
        if not isinstance(data, dict):
            stats["unparsed_batches"] += 1
            continue
        for u in data.get("units") or []:
            if not isinstance(u, dict):
                continue
            ids = resolve_labels(u.get("turns", []), b.labels)
            text = str(u.get("text", "")).strip()
            if not ids or not text:
                stats["dropped_no_provenance"] += 1
                continue
            kind = u.get("kind") if u.get("kind") in KINDS else "fact"
            ents = [str(e) for e in u.get("entities") or []][:8]
            ev = str(u.get("event_date") or "")
            ev = ev if _DATE.fullmatch(ev) else ""
            units.append(MemoryUnit(f"u{len(units)}", kind, text, ids, date_of[ids[0]], ents, ev))
    stats["extracted_units"] = len(units)
    return units, stats
