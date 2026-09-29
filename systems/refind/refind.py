"""ReFind (reimpl.) — Li et al. 2026, arXiv 2608.12888. No official code released; see
docs/reimplementation_notes.md for deviations.

Stage 1: ReAct agent (search_chatrecord / take_note / finish_search), max 4 searches, BM25 top-5 with
session-aware RRF, +-2 unit context expansion, temporal filtering, seen-session exclusion.
Stage 2 (answering) is the shared reader; collected notes are cut to B in save order (the method's own ranking).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime

from core.data.longmemeval import Question, Turn
from core.context import ContextResult, Item, fill_items, render_turn, turn_sort_key
from core.retrieval.bm25 import BM25Index

SYSTEM_PROMPT = """You are a research assistant collecting evidence from a user's conversation history. You are NOT answering — only gathering.

## Actions
search_chatrecord — Search by keywords. Stemming enabled. Previously returned sessions auto-excluded.
{"keywords": ["keyword1", "keyword2"], "top_k": 5}
take_note — Save results from the LAST search by number. Saves complete original conversations.
{"indices": [1, 3, 5]}
finish_search — Done collecting; proceed to answer.
{}

## Workflow
search → take_note → search again → ... → finish

## Rules
1. After EVERY search, review ALL results and save ANY that MIGHT be relevant. Missed info is LOST.
2. Call take_note BEFORE your next search. Previous results become inaccessible after a new search.
3. Don't answer — only collect. Try diverse keywords.
4. If no results are relevant, skip take_note and go directly to search_chatrecord or finish_search.

## Response Format
ONLY output Thought + Action + Action Input. NEVER generate "Observation:" - observations are provided by the system. Keep Thought to 1-2 sentences.

Thought: Results 1, 3, and 7 mention the topic. Saving them.
Action: take_note
Action Input: {"indices": [1, 3, 7]}

## Time Filtering
Optional date range: add date_from / date_to (YYYY/MM/DD) to search_chatrecord.
{"keywords": ["k1"], "date_from": "2023/06/01", "date_to": "2023/06/30"}
Broad first, narrow later: First search WITHOUT time filters (catches retrospective mentions). Then use time filters to find original events."""

NEXT = "Observation: {}\n\nRespond with Thought + Action + Action Input ONLY. Do NOT generate Observation yourself."


@dataclass
class Unit:
    """ReFind's turn: a user utterance paired with its assistant response."""
    idx: int
    session_id: str
    date: datetime | None
    turns: list[Turn]

    @property
    def text(self) -> str:
        return "\n".join(f"{t.role}: {t.text}" for t in self.turns)


def make_units(haystack: list[Turn]) -> list[Unit]:
    units: list[Unit] = []
    for t in haystack:
        last = units[-1] if units else None
        if last and last.session_id == t.session_id and t.role == "assistant" and last.turns[-1].role == "user":
            last.turns.append(t)
        else:
            units.append(Unit(len(units), t.session_id, t.session_date, [t]))
    return units


def parse_action(text: str) -> tuple[str | None, dict]:
    m = re.search(r"Action:\s*([a-z_]+)", text)
    if not m:
        return None, {}
    a = re.search(r"Action Input:\s*(\{.*\})", text[m.end():], re.S)
    try:
        return m.group(1), (json.loads(a.group(1)) if a else {})
    except json.JSONDecodeError:
        return m.group(1), {}


def _date(s: str | None) -> datetime | None:
    try:
        return datetime.strptime(s, "%Y/%m/%d") if s else None
    except ValueError:
        return None


class ReFindSearch:
    """Stateful search engine for one question (seen-session set persists across rounds)."""

    def __init__(self, units: list[Unit], k1=1.2, b=0.75, rrf_k=60, window=2):
        self.units, self.rrf_k, self.window = units, rrf_k, window
        self.index = BM25Index([u.text for u in units], k1=k1, b=b, stem=True)
        self.seen: set[str] = set()

    def search(self, keywords: list[str], top_k=5, date_from=None, date_to=None) -> list[list[Unit]]:
        lo, hi = _date(date_from), _date(date_to)
        hi = hi.replace(hour=23, minute=59) if hi else None
        scores = self.index.scores(" ".join(keywords))
        cand = [u for u in self.units if u.session_id not in self.seen
                and (lo is None or (u.date and u.date >= lo)) and (hi is None or (u.date and u.date <= hi))
                and scores[u.idx] > 0]
        if not cand:
            return []
        r1 = {u.idx: i for i, u in enumerate(sorted(cand, key=lambda u: (-scores[u.idx], u.idx)), 1)}
        sess: dict[str, float] = {}
        for u in cand:
            sess[u.session_id] = sess.get(u.session_id, 0.0) + scores[u.idx]
        srank = {s: i for i, s in enumerate(sorted(sess, key=lambda s: (-sess[s], s)), 1)}
        fused = sorted(cand, key=lambda u: (-(1 / (self.rrf_k + r1[u.idx]) + 1 / (self.rrf_k + srank[u.session_id])), u.idx))
        blocks = []
        for u in fused[:top_k]:
            blocks.append([v for v in self.units[max(0, u.idx - self.window): u.idx + self.window + 1]
                           if v.session_id == u.session_id])
        self.seen |= {b[0].session_id for b in blocks}
        return blocks


def format_results(blocks: list[list[Unit]]) -> str:
    if not blocks:
        return "No results."
    out = []
    for i, b in enumerate(blocks, 1):
        date = b[0].date.strftime("%Y/%m/%d") if b[0].date else "unknown"
        out.append(f"[{i}] session {b[0].session_id} ({date})\n" + "\n".join(u.text for u in b))
    return "\n\n".join(out)


class ReFind:
    name = "ReFind (reimpl.)"

    def __init__(self, services, max_searches: int = 4, top_k: int = 5, max_tokens: int = 1024):
        self.services, self.max_searches, self.top_k, self.max_tokens = services, max_searches, top_k, max_tokens

    def build_memory(self, q: Question):
        return {"units": make_units(q.haystack), "qid": q.qid}  # no LLM at construction

    def collect(self, query: str, memory, query_date) -> tuple[list[list[Unit]], dict]:
        llm = self.services.llm("query")
        engine = ReFindSearch(memory["units"])
        date = query_date.strftime("%Y/%m/%d") if query_date else "unknown"
        msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Question: {query}\nCurrent date: {date}\nBegin collecting evidence."}]
        notes: list[list[Unit]] = []
        last: list[list[Unit]] = []
        searches, bad, steps = 0, 0, []
        for _ in range(2 * self.max_searches + 2):
            reply = llm.complete(msgs, stage="query_time", qid=memory["qid"], max_tokens=self.max_tokens).text
            action, args = parse_action(reply)
            steps.append({"action": action, "args": args})
            msgs.append({"role": "assistant", "content": reply})
            if action == "finish_search":
                break
            if action == "search_chatrecord" and searches < self.max_searches:
                kws = args.get("keywords") or []
                kws = [kws] if isinstance(kws, str) else [str(k) for k in kws]
                last = engine.search(kws, int(args.get("top_k", self.top_k)), args.get("date_from"), args.get("date_to"))
                searches += 1
                obs = format_results(last)
                if searches == self.max_searches:
                    obs += "\n\n(Search limit reached: save anything relevant with take_note, then finish_search.)"
            elif action == "search_chatrecord":
                obs = "Search limit reached. Use take_note on the last results or finish_search."
            elif action == "take_note":
                idx = [i for i in args.get("indices", []) if isinstance(i, int) and 1 <= i <= len(last)]
                notes += [last[i - 1] for i in idx]
                obs = f"Saved {len(idx)} result(s)."
            else:
                bad += 1
                obs = "Invalid action. Use search_chatrecord, take_note or finish_search with a JSON Action Input."
            msgs.append({"role": "user", "content": NEXT.format(obs)})
        return notes, {"steps": steps, "searches": searches, "invalid_actions": bad}

    def build_context(self, query, memory, token_budget, query_date=None) -> ContextResult:
        notes, trace = self.collect(query, memory, query_date)
        # Budget cut at message granularity, in save order = the method's own ranking.
        # Within a note, the matched (center) unit comes first, then neighbours by distance.
        seen, items = set(), []
        for block in notes:
            center = block[len(block) // 2].idx if block else 0
            for u in sorted(block, key=lambda u: (abs(u.idx - center), u.idx)):
                if u.idx in seen:
                    continue
                seen.add(u.idx)
                # one item per message, so a long assistant reply does not push out its user turn
                items += [Item(render_turn(t), [t.turn_id], turn_sort_key(t)) for t in u.turns]
        r = fill_items(items, token_budget, query_date)
        r.trace.update(trace, n_notes=len(notes), n_messages_collected=len(items), messages_cut=len(items) - len(r.trace["rank_of_chosen"]))
        return r
