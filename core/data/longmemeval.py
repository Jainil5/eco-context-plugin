"""LongMemEval loader -> normalized Turn / Question schema (brief §3).

Source file format (official cleaned release, xiaowu0162/longmemeval-cleaned):
  question_id, question_type, question, question_date, answer (str | int),
  answer_session_ids, haystack_session_ids, haystack_dates, haystack_sessions
  haystack_sessions[i][j] = {"role", "content", "has_answer"?}
Some haystacks repeat a session id (see normalize).
`has_answer` only appears on turns inside answer sessions; elsewhere it is None.
Abstention questions have question_id ending in "_abs".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class Turn:
    turn_id: str          # "<session_id>:<index>", unique within a question's haystack
    session_id: str
    session_date: datetime | None
    role: str             # "user" | "assistant"
    text: str
    has_answer: bool | None


@dataclass
class Question:
    qid: str
    qtype: str            # LongMemEval category; abstention is flagged separately
    question: str
    question_date: datetime | None
    answer: str
    haystack: list[Turn]
    evidence_turn_ids: list[str]
    is_abstention: bool = False
    evidence_session_ids: list[str] = field(default_factory=list)

    @property
    def stratum(self) -> str:
        return f"{self.qtype}{'_abs' if self.is_abstention else ''}"


def parse_date(s: str | None) -> datetime | None:
    """'2023/05/20 (Sat) 02:21' -> datetime(2023, 5, 20, 2, 21)."""
    if not s:
        return None
    date, _, rest = s.partition(" (")
    time = rest.partition(") ")[2]
    return datetime.strptime(f"{date} {time}".strip(), "%Y/%m/%d %H:%M" if time else "%Y/%m/%d")


def normalize(raw: dict) -> Question:
    sids, dates, sessions = raw["haystack_session_ids"], raw["haystack_dates"], raw["haystack_sessions"]
    if not (len(sids) == len(dates) == len(sessions)):
        raise ValueError(f"{raw['question_id']}: haystack field lengths differ")
    # The official file repeats a few sessions (same id + text, different date) in 13 haystacks.
    # Keep every occurrence; later occurrences get "#2", "#3" so turn ids stay unique.
    seen: dict[str, int] = {}
    turns: list[Turn] = []
    for sid, date, sess in zip(sids, dates, sessions):
        seen[sid] = seen.get(sid, 0) + 1
        if seen[sid] > 1:
            sid = f"{sid}#{seen[sid]}"
        d = parse_date(date)
        for j, t in enumerate(sess):
            turns.append(Turn(turn_id=f"{sid}:{j}", session_id=sid, session_date=d, role=t["role"],
                              text=t["content"], has_answer=t.get("has_answer")))
    return Question(
        qid=raw["question_id"], qtype=raw["question_type"], question=raw["question"],
        question_date=parse_date(raw.get("question_date")), answer=str(raw["answer"]),
        haystack=turns, evidence_turn_ids=[t.turn_id for t in turns if t.has_answer],
        is_abstention=raw["question_id"].endswith("_abs"),
        evidence_session_ids=list(raw.get("answer_session_ids", [])),
    )


def load_longmemeval(path: str | Path) -> list[Question]:
    return [normalize(r) for r in json.loads(Path(path).read_text())]
