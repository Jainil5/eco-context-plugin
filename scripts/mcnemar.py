"""Exact McNemar test for one system against every other system in ONE logged run (paired binary outcomes).
Usage: python scripts/mcnemar.py results/runs/<run_id> [system=eco_spans] [metric=correct] [file=predictions.jsonl]
       metric=correct is the QA run; a graded metric is tested on the sign of the per-question difference.
Writes results/tables/<run_id>_mcnemar_<system>.csv. The paired bootstrap in paired_compare.py answers "how big and
how sure", this answers "is the direction more than a coin flip": with n discordant questions and no true difference
each side is a fair coin, so the exact two-sided p is 2 x P(Bin(n, 1/2) >= max(b, c)). At the sample sizes these runs
use, the bootstrap CI excluding zero and McNemar p < 0.05 are not the same statement, and both are reported.
"""
import csv
import json
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

run = Path(sys.argv[1])
run = run if run.is_absolute() else REPO_ROOT / run
system = sys.argv[2] if len(sys.argv) > 2 else "eco_spans"
metric = sys.argv[3] if len(sys.argv) > 3 else "correct"
preds_file = sys.argv[4] if len(sys.argv) > 4 else "predictions.jsonl"


def exact_p(b: int, c: int) -> float:
    """Two-sided exact McNemar: 2 x P(Bin(b + c, 1/2) >= max(b, c)), capped at 1."""
    n, m = b + c, max(b, c)
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(m, n + 1)) / 2 ** n
    return min(1.0, 2 * tail)


vals = defaultdict(dict)  # (system, budget) -> {qid: value}
for line in open(run / preds_file):
    p = json.loads(line)
    if p.get(metric) is not None:
        vals[(p["system"], p["budget"])][p["qid"]] = float(p[metric])

if system not in {s for s, _ in vals}:
    sys.exit(f"{system!r} has no {metric!r} values in {run.name}; systems here: {sorted({s for s, _ in vals})}")

rows = []
for b in sorted({b for _, b in vals}):
    for base in sorted({s for s, bb in vals if bb == b and s != system}):
        qids = sorted(set(vals[(system, b)]) & set(vals[(base, b)]))
        if not qids:
            continue
        d = [vals[(system, b)][q] - vals[(base, b)][q] for q in qids]
        up, down = sum(x > 0 for x in d), sum(x < 0 for x in d)
        rows.append({"system": system, "baseline": base, "budget": b, "metric": metric, "n": len(d),
                     "better": up, "worse": down, "discordant": up + down,
                     "p_exact": round(exact_p(up, down), 4), "run_id": run.name})

name = f"{run.name}_mcnemar_{system}" + ("" if metric == "correct" else f"_{metric}")
out = REPO_ROOT / "results" / "tables" / f"{name}.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"wrote {out.relative_to(REPO_ROOT)} ({len(rows)} rows)")
for r in rows:
    print(f"{r['budget']:>6}  {r['baseline']:<12}  {r['better']:>2} up / {r['worse']:>2} down  "
          f"n_disc={r['discordant']:>2}  p_exact={r['p_exact']:.3f}")
