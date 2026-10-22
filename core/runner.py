"""System x budget runner. For each (question, system): build memory once, build a context per budget, then
  eval="recall": evidence recall only (no reader/judge calls), or
  eval="qa":     shared reader -> shared judge (LongMemEval prompt).
Everything is logged to one run dir: predictions.jsonl, costs.jsonl, errors.jsonl, traces/<system>/."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from core.context import Services
from core.data.longmemeval import Question
from core.eval.judge import judge
from core.eval.reader import read
from core.eval.recall import answer_in_context, content_recall, evidence_recall
from core.llm.cache import DiskCache
from core.llm.client import EmbeddingClient, LLMClient
from core.llm.cost import CostLogger, Pricing
from core.llm.tokenizer import get_tokenizer
from systems import build_system

SLOW_FIRST = {"eco_facts": 0, "memorycpt": 0, "refind": 1}  # long jobs first, fast systems fill the gaps


class Runner:
    def __init__(self, cfg: dict, run_dir: Path, chat_model, judge_model, embeddings, eval_mode: str = "recall"):
        """chat_model: one model, or a list (pool). With a pool, each question is pinned to one pool member
        by a stable hash of its id, so all systems use the same provider on the same question."""
        assert eval_mode in ("recall", "qa"), eval_mode
        self.cfg, self.run_dir, self.eval_mode = cfg, run_dir, eval_mode
        self.pool = chat_model if isinstance(chat_model, list) else [chat_model]
        self.cache = DiskCache(cfg["paths"]["cache"])
        self.logger = CostLogger(run_dir / "costs.jsonl", Pricing(cfg.get("pricing", {})))
        self.judge_model, self.embeddings = judge_model, embeddings
        # params every chat call in this run carries (e.g. reasoning_effort); they also enter the cache key
        self.llm_defaults = dict(cfg.get("llm_call_params") or {})
        self._lock = threading.Lock()
        self._qid_locks: dict[str, threading.Lock] = {}

    def qid_lock(self, qid: str) -> threading.Lock:
        """One lock per question: every system building memory for the same haystack embeds the same texts,
        and concurrent first-timers all miss the embedding cache at once and pay for the same work N times."""
        with self._lock:
            if qid not in self._qid_locks:
                self._qid_locks[qid] = threading.Lock()
            return self._qid_locks[qid]

    def chat_for(self, qid: str):
        return self.pool[int(hashlib.md5(qid.encode()).hexdigest(), 16) % len(self.pool)]

    def services(self, system: str, chat_model) -> Services:
        d = self.llm_defaults
        chat = LLMClient(chat_model, self.cache, self.logger, method=system, defaults=d) if chat_model is not None else None
        emb = EmbeddingClient(self.embeddings, self.cache, self.logger, method=system) if self.embeddings else None
        return Services(llms={"extraction": chat, "query": chat} if chat else {}, embedder=emb)

    def _write(self, name: str, row: dict) -> None:
        with self._lock, open(self.run_dir / name, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

    def run_one(self, q: Question, system: str, budgets: list[int]) -> list[dict]:
        chat_model = self.chat_for(q.qid)
        method = build_system(system, self.services(system, chat_model), self.cfg.get("construction_parallel", 4))
        tok = get_tokenizer()
        t0 = time.perf_counter()
        with self.qid_lock(q.qid):  # index/embed the shared haystack once, not once per system
            memory = method.build_memory(q)
        t_build = time.perf_counter() - t0
        gold = set(q.evidence_turn_ids)
        gold_texts = [t.text for t in q.haystack if t.turn_id in gold]
        rows = []
        for b in budgets:
            t1 = time.perf_counter()
            res = method.build_context(q.question, memory, b, q.question_date)
            t_ctx = time.perf_counter() - t1
            assert res.n_tokens <= b, f"{system}: context {res.n_tokens} > budget {b}"
            row = {"system": system, "system_name": method.name, "budget": b, "qid": q.qid, "qtype": q.qtype,
                   "is_abstention": q.is_abstention, "view_tokens": sum(tok.count(t.text) for t in q.haystack),
                   "view_turns": len(q.haystack), "n_tokens": res.n_tokens,
                   "evidence_recall": evidence_recall(res.source_turn_ids, q.evidence_turn_ids),
                   "verbatim_recall": evidence_recall(res.verbatim_turn_ids, q.evidence_turn_ids),
                   "content_recall": content_recall(res.context, gold_texts),
                   "answer_in_ctx": answer_in_context(res.context, q.answer),
                   "latency_build_s": t_build, "latency_context_s": t_ctx}
            if self.eval_mode == "qa":
                reader = LLMClient(chat_model, self.cache, self.logger, method=system, defaults=self.llm_defaults)
                judge_c = LLMClient(self.judge_model, self.cache, self.logger, method=system,
                                    defaults=self.llm_defaults)
                hyp = read(reader, q.question, res.context, q.question_date, qid=q.qid,
                           max_tokens=self.cfg.get("reader_max_tokens", 1024))
                ok, verdict = judge(judge_c, q, hyp, max_tokens=self.cfg.get("judge_max_tokens", 10))
                row.update(llm=reader.model, judge_model=judge_c.model, correct=ok, hypothesis=hyp,
                           judge_reply=verdict, gold=q.answer)
            rows.append(row)
            self._write("predictions.jsonl", row)
            tdir = self.run_dir / "traces" / system
            tdir.mkdir(parents=True, exist_ok=True)
            (tdir / f"{q.qid}_{b}.json").write_text(json.dumps(
                {"context": res.context, "source_turn_ids": res.source_turn_ids,
                 "verbatim_turn_ids": res.verbatim_turn_ids, "trace": res.trace}, default=str))
        return rows

    def run(self, questions: list[Question], systems: list[str], budgets: list[int], workers: int = 4,
            progress=print) -> None:
        # system-major, so the jobs in flight at any moment are mostly *different* questions: with a per-question
        # build lock, a question-major queue would park every worker on one haystack while its systems line up.
        jobs = sorted(((q, s) for s in systems for q in questions), key=lambda j: SLOW_FIRST.get(j[1], 2))
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(self.run_one, q, s, budgets): (q.qid, s) for q, s in jobs}
            for f in as_completed(futs):
                qid, s = futs[f]
                done += 1
                try:
                    rows = f.result()
                    if self.eval_mode == "qa":
                        msg = " ".join(f"B={r['budget']}:{'ok' if r['correct'] else 'x'}" for r in rows)
                    else:
                        msg = " ".join(f"B={r['budget']}:{r['verbatim_recall'] if r['verbatim_recall'] is None else round(r['verbatim_recall'], 2)}"
                                       for r in rows)
                    progress(f"[{done}/{len(jobs)}] {s:<10} {qid:<14} {msg}")
                except Exception as e:
                    self._write("errors.jsonl", {"qid": qid, "system": s, "error": repr(e),
                                                 "traceback": traceback.format_exc()})
                    progress(f"[{done}/{len(jobs)}] {s:<10} {qid:<14} ERROR {e!r}"[:200])
