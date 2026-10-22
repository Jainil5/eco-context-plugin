"""Measure the premise the whole method rests on: how redundant is what each system puts in the context?

The README says top-k "fills the budget with the highest-scoring turns, which are often near-duplicates of each
other", and that facility location is the fix. That has been asserted, never measured. Every run already logs the
exact context string in traces/, and each line is one selected item, so redundancy is computable from data that
exists: strip the header, embed the text (a cache hit, since the run embedded it), take pairwise cosines.

Reported per (system, budget): how many items were selected, the mean pairwise cosine between them, the share of
pairs at or above ECO's similarity floor (0.7, the threshold at which ECO treats two items as covering each other),
and — count-independent, so it compares across budgets honestly — how often a context contains at least one such
pair.

Usage: python scripts/redundancy.py results/runs/<run_id> [floor=0.7]
Writes results/tables/<run_id>_redundancy.csv and per-context rows in results/runs/<run_id>/redundancy.jsonl.
"""
from __future__ import annotations

import csv
import json
import re
import sys
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore", message="No price")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from core.config import REPO_ROOT, load_config  # noqa: E402
from core.eval.compare import bootstrap_ci  # noqa: E402
from core.llm.cache import DiskCache  # noqa: E402
from core.llm.client import EmbeddingClient  # noqa: E402
from core.llm.cost import CostLogger, Pricing  # noqa: E402
from core.llm.models import get_embeddings  # noqa: E402

run = Path(sys.argv[1])
run = run if run.is_absolute() else REPO_ROOT / run
FLOOR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.7
cfg = load_config(run / "config.yaml")

# render_turn writes "[2026-01-02 | e6:3] user: text"; span continuations write "[e6:3] text". Stripping the
# leading bracket group and an optional role leaves the raw item text, which is the string that was embedded.
HEADER = re.compile(r"^\s*\[[^\]]*\]\s*(?:(?:user|assistant)\s*:\s*)?")


def items_of(context: str) -> list[str]:
    out = []
    for ln in context.split("\n"):
        if not ln.strip() or ln.startswith("Current date:"):
            continue
        text = HEADER.sub("", ln, count=1).strip()
        if text:
            out.append(text)
    return out


emb = EmbeddingClient(get_embeddings(), DiskCache(REPO_ROOT / cfg["paths"]["cache"]),
                      CostLogger(run / "redundancy_costs.jsonl", Pricing(cfg.get("pricing", {}))),
                      method="redundancy")

groups: dict[tuple, list[float]] = defaultdict(list)   # (system, budget) -> [mean pairwise cos]
counts: dict[tuple, list[float]] = defaultdict(list)   # (system, budget) -> [share of pairs >= floor]
dupes: dict[tuple, list[float]] = defaultdict(list)    # (system, budget) -> [1 if any pair >= floor else 0]
n_items: dict[tuple, list[int]] = defaultdict(list)
per_ctx: list[dict] = []                         # one row per context, so paired comparisons are possible later
scored, no_trace = 0, 0
for line in (run / "predictions.jsonl").read_text().splitlines():
    p = json.loads(line)
    tp = run / "traces" / p["system"] / f"{p['qid']}_{p['budget']}.json"
    if not tp.exists():
        no_trace += 1
        continue
    texts = items_of(json.loads(tp.read_text())["context"])
    key = {"system": p["system"], "qid": p["qid"], "qtype": p.get("qtype"), "budget": p["budget"],
           "n_items": len(texts)}
    if len(texts) < 2:
        per_ctx.append({**key, "mean_pairwise_cos": None, "share_pairs_ge_floor": None, "has_dup": None})
        scored += 1
        n_items[(p["system"], p["budget"])].append(len(texts))
        continue
    V = np.asarray(emb.embed(texts, stage="query_time", qid=p["qid"]), dtype=np.float32)
    V = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = V @ V.T
    iu = np.triu_indices(len(texts), 1)
    pairs = S[iu]
    groups[(p["system"], p["budget"])].append(float(pairs.mean()))
    counts[(p["system"], p["budget"])].append(float((pairs >= FLOOR).mean()))
    dupes[(p["system"], p["budget"])].append(float((pairs >= FLOOR).any()))
    per_ctx.append({**key, "mean_pairwise_cos": round(float(pairs.mean()), 6),
                    "share_pairs_ge_floor": round(float((pairs >= FLOOR).mean()), 6),
                    "has_dup": int((pairs >= FLOOR).any())})
    n_items[(p["system"], p["budget"])].append(len(texts))
    scored += 1

(run / "redundancy.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_ctx))

rows = []
for (s, b) in sorted(groups.keys() | set(counts) | set(n_items)):
    g, c, d, n = groups.get((s, b), []), counts.get((s, b), []), dupes.get((s, b), []), n_items.get((s, b), [])
    lo, hi = bootstrap_ci(g) if g else (float("nan"),) * 2
    dlo, dhi = bootstrap_ci(d) if d else (float("nan"),) * 2
    rows.append({"system": s, "budget": b, "n_contexts": len(n), "avg_items": round(sum(n) / len(n), 2) if n else "",
                 "mean_pairwise_cos": round(sum(g) / len(g), 4) if g else "",
                 "ci_low": round(lo, 4) if g else "", "ci_high": round(hi, 4) if g else "",
                 "share_pairs_at_or_above_floor": round(sum(c) / len(c), 4) if c else "",
                 "contexts_with_a_duplicate_pair": round(sum(d) / len(d), 4) if d else "",
                 "dup_ci_low": round(dlo, 4) if d else "", "dup_ci_high": round(dhi, 4) if d else "",
                 "floor": FLOOR, "run_id": run.name})
table = REPO_ROOT / "results" / "tables" / f"{run.name}_redundancy.csv"
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"{scored} contexts scored ({no_trace} without a trace) -> {table.relative_to(REPO_ROOT)}")
print(f"{'system':<13}{'B':>6}{'n':>5}{'items':>7}{'mean cos':>10}{'pairs>=f':>10}{'ctx w/ dup':>11}")
for r in rows:
    print(f"{r['system']:<13}{r['budget']:>6}{r['n_contexts']:>5}{r['avg_items'] or '-':>7}"
          f"{r['mean_pairwise_cos'] or '-':>10}{r['share_pairs_at_or_above_floor'] or '-':>10}"
          f"{r['contexts_with_a_duplicate_pair'] or '-':>11}")
