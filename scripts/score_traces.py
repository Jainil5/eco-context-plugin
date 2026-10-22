"""Recompute context-derived metrics for a logged run from its saved traces. No LLM, no re-embedding.

content_recall (how much gold sentence text reached the reader) and answer_in_ctx (whether the gold answer itself
reached the reader) are pure functions of the rendered context string, which every run already stores in
traces/<system>/<qid>_<budget>.json. So a metric decided on after a run launched can still be scored on it,
exactly as it was logged, rather than paying for the run again.

Usage:
  python scripts/score_traces.py results/runs/<run_id>
Writes:
  results/runs/<run_id>/trace_metrics.jsonl        one enriched row per (system, budget, qid)
  results/tables/<run_id>_trace_metrics.csv        system x budget x qtype means with 95% bootstrap CIs
Then, for paired differences against the strongest baseline per budget:
  python scripts/paired_compare.py results/runs/<run_id> eco content_recall trace_metrics.jsonl
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import REPO_ROOT  # noqa: E402
from core.data.longmemeval import load_longmemeval  # noqa: E402
from core.eval.compare import bootstrap_ci  # noqa: E402
from core.eval.recall import answer_in_context, content_recall  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
run = run if run.is_absolute() else REPO_ROOT / run
cfg = yaml.safe_load((run / "config.yaml").read_text())
qs = {q.qid: q for q in load_longmemeval(REPO_ROOT / cfg["data"]["longmemeval_s"])}

preds = [json.loads(l) for l in (run / "predictions.jsonl").read_text().splitlines() if l.strip()]
rows, missing = [], 0
for p in preds:
    tp = run / "traces" / p["system"] / f"{p['qid']}_{p['budget']}.json"
    if not tp.exists():
        missing += 1
        continue
    q = qs[p["qid"]]
    ctx = json.loads(tp.read_text())["context"]
    gold = [t.text for t in q.haystack if t.turn_id in set(q.evidence_turn_ids)]
    rows.append({**{k: p[k] for k in ("system", "budget", "qid", "qtype", "n_tokens", "verbatim_recall",
                                      "evidence_recall")},
                 "content_recall": content_recall(ctx, gold),
                 "answer_in_ctx": answer_in_context(ctx, q.answer)})

(run / "trace_metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")

groups: dict[tuple, list[dict]] = defaultdict(list)
for r in rows:
    groups[(r["system"], r["budget"], "ALL")].append(r)
    groups[(r["system"], r["budget"], r["qtype"])].append(r)

out = []
for (s, b, qt), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2] != "ALL", x[0][2])):
    row = {"system": s, "budget": b, "qtype": qt, "n": len(rs)}
    for key in ("verbatim_recall", "content_recall", "answer_in_ctx"):
        xs = [float(r[key]) for r in rs if r.get(key) is not None]
        lo, hi = bootstrap_ci(xs)
        row[key] = round(sum(xs) / len(xs), 4) if xs else ""
        row[f"{key}_ci_low"] = round(lo, 4) if xs else ""
        row[f"{key}_ci_high"] = round(hi, 4) if xs else ""
    row["avg_tokens"] = round(sum(r["n_tokens"] for r in rs) / len(rs), 1)
    row["run_id"] = run.name
    out.append(row)

table = REPO_ROOT / "results" / "tables" / f"{run.name}_trace_metrics.csv"
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"{len(rows)} rows scored ({missing} without a trace) -> {table.relative_to(REPO_ROOT)}")
print(f"{'system':<12}{'B':>6}{'n':>5}{'verbatim':>10}{'content':>9}{'answer':>8}{'tokens':>8}")
for r in out:
    if r["qtype"] != "ALL":
        continue
    print(f"{r['system']:<12}{r['budget']:>6}{r['n']:>5}{r['verbatim_recall']:>10}{r['content_recall']:>9}"
          f"{r['answer_in_ctx']:>8}{r['avg_tokens']:>8}")
