"""Did the ECO mechanisms actually fire? Aggregates the objective traces of a logged run.

Every mechanism writes what it changed into the per-(question, budget) trace (superseded_pairs, n_candidate_sessions,
facet_terms, session_mass_share), so this answers "was there anything for the mechanism to do on this question"
without re-running anything. It is the check that keeps a null result honest: a mechanism that never fires is not
evidence that the mechanism is useless, it is evidence the testbed cannot see it.

Usage:
  python scripts/mechanisms.py results/runs/<run_id>
Writes:
  results/tables/<run_id>_mechanisms.csv
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import REPO_ROOT  # noqa: E402
from core.eval.compare import bootstrap_ci  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs")
run = run if run.is_absolute() else REPO_ROOT / run
metrics = {}
if (run / "trace_metrics.jsonl").exists():
    for line in (run / "trace_metrics.jsonl").read_text().splitlines():
        r = json.loads(line)
        metrics[(r["system"], r["budget"], r["qid"])] = r

groups: dict[tuple, list[dict]] = defaultdict(list)
for tdir in sorted((run / "traces").iterdir()):
    for f in sorted(tdir.glob("*.json")):
        system, rest = f.parent.name, f.stem
        qid, budget = rest.rsplit("_", 1)
        t = json.loads(f.read_text())["trace"]
        m = metrics.get((system, int(budget), qid), {})
        groups[(int(budget), system)].append({**t, "answer_in_ctx": m.get("answer_in_ctx"),
                                              "content_recall": m.get("content_recall"),
                                              "n_tokens": m.get("n_tokens", t.get("cost"))})

out = []
for (b, s), rs in sorted(groups.items(), key=lambda x: (x[0][0], -1 if x[0][1] == "eco_spans" else 0, x[0][1])):
    row = {"budget": b, "system": s, "n": len(rs)}
    for key, label in (("answer_in_ctx", "answer"), ("content_recall", "content"), ("n_tokens", "tokens")):
        xs = [float(r[key]) for r in rs if r.get(key) is not None]
        if key == "answer_in_ctx" and xs:
            lo, hi = bootstrap_ci(xs)
            row["answer_ci"] = f"{lo:.3f}..{hi:.3f}"
        row[label] = round(sum(xs) / len(xs), 4) if xs else ""
    for key, label in (("n_selected", "spans"), ("n_sessions_selected", "sessions_in_ctx"),
                       ("n_candidate_sessions", "sessions_in_pool"), ("superseded_pairs", "superseded_pairs")):
        xs = [float(r[key]) for r in rs if r.get(key) is not None]
        row[label] = round(sum(xs) / len(xs), 2) if xs else ""
    row["facet_queries"] = round(sum(1 for r in rs if r.get("facet_terms")) / len(rs), 3)
    row["breadth_queries"] = round(sum(1 for r in rs if float(r.get("n_candidate_sessions") or 1) > 1) / len(rs), 3)
    out.append(row)

table = REPO_ROOT / "results" / "tables" / f"{run.name}_mechanisms.csv"
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
print(f"{sum(len(v) for v in groups.values())} traces -> {table.relative_to(REPO_ROOT)}")
cols = ["budget", "system", "n", "answer", "answer_ci", "content", "tokens", "spans",
        "sessions_in_ctx", "sessions_in_pool", "breadth_queries", "superseded_pairs", "facet_queries"]
widths = {c: max(len(c), *(len(str(r[c])) for r in out)) for c in cols}
print("".join(str(c).ljust(widths[c] + 1) for c in cols))
for r in out:
    print("".join(str(r[c]).ljust(widths[c] + 1) for c in cols))
