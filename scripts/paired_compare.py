"""Paired comparison of one system against the others in ONE logged run (same questions, per-question differences).
Usage: python scripts/paired_compare.py results/runs/<run_id> [system=eco] [metric=verbatim_recall]
Writes results/tables/<run_id>_paired_<system>.csv: per budget and baseline, the mean difference with a 95% paired
bootstrap CI (5000 resamples, seed 0) and the number of questions where the system is better / worse.
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

run = Path(sys.argv[1])
run = run if run.is_absolute() else REPO_ROOT / run
system = sys.argv[2] if len(sys.argv) > 2 else "eco"
metric = sys.argv[3] if len(sys.argv) > 3 else "verbatim_recall"
EXCLUDE = {system, "oracle", "recent"}  # oracle is an upper bound, recent a floor

vals = defaultdict(dict)  # (system, budget) -> {qid: value}
for line in open(run / "predictions.jsonl"):
    p = json.loads(line)
    if p.get(metric) is not None:
        vals[(p["system"], p["budget"])][p["qid"]] = float(p[metric])

rows = []
for b in sorted({b for _, b in vals}):
    for base in sorted({s for s, bb in vals if bb == b and s not in EXCLUDE}):
        qids = sorted(set(vals[(system, b)]) & set(vals[(base, b)]))
        diff = [vals[(system, b)][q] - vals[(base, b)][q] for q in qids]
        rng = random.Random(0)
        boots = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(5000))
        rows.append({"system": system, "baseline": base, "budget": b, "metric": metric, "n": len(diff),
                     "mean_diff": round(sum(diff) / len(diff), 4), "ci_low": round(boots[125], 4),
                     "ci_high": round(boots[4874], 4), "better": sum(d > 0 for d in diff),
                     "worse": sum(d < 0 for d in diff), "run_id": run.name})

out = REPO_ROOT / "results" / "tables" / f"{run.name}_paired_{system}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
