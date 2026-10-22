"""Budget efficiency: how many tokens each system needs to reach a target, not what it reaches at four budgets.

Point comparisons at B in {250, 500, 1000, 2000} hide the thing a budget paper is actually about — the price of a
given recall level. This script inverts the curve: for every (system, question) it finds the smallest logged
context that reaches the target metric, interpolating linearly in log(tokens) between the two bracketing budgets,
then reports the mean over questions. A system that needs 620 tokens where another needs 1,400 is the finding.

Questions that never reach the target at any logged budget are counted and excluded from the mean, so the number
is conditioned on "reachable at all"; the reachable share is reported next to it so nobody has to guess.

Usage: python scripts/efficiency.py results/runs/<run_id> [metrics=content_recall,verbatim_recall] [targets=0.5,0.8,0.9]
Writes results/tables/<run_id>_efficiency.csv — one block of rows per metric, so the metrics stay in one table and a
second invocation does not overwrite the first.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402
from core.eval.compare import bootstrap_ci  # noqa: E402

run = Path(sys.argv[1])
run = run if run.is_absolute() else REPO_ROOT / run
metrics = [m.strip() for m in (sys.argv[2] if len(sys.argv) > 2 else "content_recall,verbatim_recall").split(",")]
targets = [float(t) for t in (sys.argv[3] if len(sys.argv) > 3 else "0.5,0.8,0.9").split(",")]
src = run / ("trace_metrics.jsonl" if (run / "trace_metrics.jsonl").exists() else "predictions.jsonl")
records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
missing = [m for m in metrics if m not in records[0]]
if missing:
    sys.exit(f"{src.name} has no column(s) {missing}; metrics available: {sorted(records[0])}")

by_q: dict[str, dict[tuple, list[tuple[int, int, float]]]] = {m: defaultdict(list) for m in metrics}
for r in records:
    for m in metrics:
        v = r.get(m)
        if v is not None:
            by_q[m][(r["system"], r["qid"])].append((int(r["budget"]), int(r["n_tokens"]), float(v)))


def min_tokens_for(points: list[tuple[int, int, float]], target: float) -> float | None:
    """Smallest token count where the metric reaches `target`, interpolated in log(tokens)."""
    pts = sorted({b: (t, v) for b, t, v in points}.items())
    if not pts or pts[0][1][1] >= target:  # already there at the smallest budget
        return float(pts[0][1][0]) if pts else None
    for i in range(1, len(pts)):
        (b0, (t0, v0)), (b1, (t1, v1)) = pts[i - 1], pts[i]
        if v1 >= target:
            if v1 == v0 or t0 <= 0:
                return float(t1)
            w = (target - v0) / (v1 - v0)
            return float(math.exp(math.log(t0) + w * (math.log(t1) - math.log(t0))))
    return None


rows = []
for metric in metrics:
    curves = by_q[metric]
    for s in sorted({sy for sy, _ in curves}):
        qs = sorted({qid for (sy, qid) in curves if sy == s})
        for target in targets:
            vals = [min_tokens_for(curves[(s, qid)], target) for qid in qs]
            got = [v for v in vals if v is not None]
            lo, hi = bootstrap_ci(got) if got else (float("nan"), float("nan"))
            rows.append({"system": s, "metric": metric, "target": target, "n_questions": len(qs),
                        "reachable": round(len(got) / len(qs), 3) if qs else "",
                        "median_tokens_to_target": round(sorted(got)[len(got) // 2], 1) if got else "",
                        "mean_tokens_to_target": round(sum(got) / len(got), 1) if got else "",
                        "ci_low": round(lo, 1), "ci_high": round(hi, 1), "run_id": run.name})

table = REPO_ROOT / "results" / "tables" / f"{run.name}_efficiency.csv"
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"{src.name}: {len(by_q[metrics[0]])} (system, question) curves x {len(metrics)} metrics "
      f"-> {table.relative_to(REPO_ROOT)}")
for metric in metrics:
    curves = by_q[metric]
    systems = sorted({sy for sy, _ in curves})
    print(f"\n--- tokens to reach {metric} ---")
    print(f"{'system':<12}" + "".join(f"{'B@' + str(t):>14}{'reach':>7}" for t in targets))
    for s in systems:
        line = f"{s:<12}"
        for target in targets:
            r = next(x for x in rows if x["metric"] == metric and x["system"] == s and x["target"] == target)
            line += f"{r['mean_tokens_to_target'] or '-':>14}{r['reachable'] or '-':>7}"
        print(line)
