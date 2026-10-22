"""Paired comparison of one system against the others in ONE logged run (same questions, per-question differences).
Usage: python scripts/paired_compare.py results/runs/<run_id> [system=eco] [metric=verbatim_recall]
       [predictions file=predictions.jsonl]  # pass trace_metrics.jsonl to pair on a rescoring run
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
preds_file = sys.argv[4] if len(sys.argv) > 4 else "predictions.jsonl"
EXCLUDE = {system, "oracle", "recent"}  # oracle is an upper bound, recent a floor

vals = defaultdict(dict)  # (system, budget) -> {qid: value}
for line in open(run / preds_file):
    p = json.loads(line)
    if p.get(metric) is not None:
        vals[(p["system"], p["budget"])][p["qid"]] = float(p[metric])

rows = []
if system not in {s for s, _ in vals}:
    sys.exit(f"{system!r} has no {metric!r} values in {run.name}; systems here: {sorted({s for s, _ in vals})}")
for b in sorted({b for _, b in vals}):
    for base in sorted({s for s, bb in vals if bb == b and s not in EXCLUDE}):
        qids = sorted(set(vals[(system, b)]) & set(vals[(base, b)]))
        if not qids:
            continue                       # this baseline does not cover this budget in this run
        diff = [vals[(system, b)][q] - vals[(base, b)][q] for q in qids]
        rng = random.Random(0)
        boots = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(5000))
        rows.append({"system": system, "baseline": base, "budget": b, "metric": metric, "n": len(diff),
                     "mean_diff": round(sum(diff) / len(diff), 4), "ci_low": round(boots[125], 4),
                     "ci_high": round(boots[4874], 4), "better": sum(d > 0 for d in diff),
                     "worse": sum(d < 0 for d in diff), "run_id": run.name})

suffix = "" if preds_file == "predictions.jsonl" else f"_{Path(preds_file).stem}"
name = f"{run.name}_paired_{system}{suffix}" + ("" if metric == "verbatim_recall" else f"_{metric}")
out = REPO_ROOT / "results" / "tables" / f"{name}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
