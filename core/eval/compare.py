"""Method comparison table built only from logged runs in results/runs/ (rule 1).

Each run dir is expected to contain:
  predictions.jsonl: {"method", "budget", "qid", "qtype", "correct": bool,
                      "n_tokens", "evidence_recall": float | null}
  costs.jsonl:       CallRecord lines from core.llm.cost (method, qid, stage, tokens, usd)
Methods absent from results simply do not appear; nothing is filled in by hand.
"""
from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

COLUMNS = [
    "method", "budget", "qtype", "n_questions",
    "accuracy", "ci95_low", "ci95_high", "evidence_recall",
    "avg_reader_input_tokens", "construction_tokens_per_q", "query_time_tokens_per_q",
    "system_usd_per_q", "acc_per_1k_reader_tokens",
    "rank_at_budget", "best_at_budget", "delta_vs_best",
    "run_ids",
]


def _read_jsonl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def bootstrap_ci(xs: list[float], n: int = 2000, seed: int = 0) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n) - 1]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_runs(runs_dir: str | Path) -> tuple[list[dict], list[dict]]:
    preds, costs = [], []
    for d in sorted(Path(runs_dir).glob("*/")):
        for r in _read_jsonl(d / "predictions.jsonl"):
            preds.append({**r, "run_id": d.name})
        costs += _read_jsonl(d / "costs.jsonl")
    return preds, costs


def build_rows(preds: list[dict], costs: list[dict]) -> list[dict]:
    # per (method, qid) cost by stage; construction is per question, independent of budget
    stage_tok: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    stage_usd: dict[tuple, float] = defaultdict(float)
    for c in costs:
        if c.get("stage") == "judge" or not c.get("method"):
            continue
        k = (c["method"], c.get("qid"))
        stage_tok[k][c["stage"]] += c["input_tokens"] + c["output_tokens"]
        stage_usd[k] += c.get("usd") or 0.0

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for p in preds:
        groups[(p["method"], p["budget"], "all")].append(p)
        groups[(p["method"], p["budget"], p["qtype"])].append(p)

    rows = []
    for (method, budget, qtype), ps in groups.items():
        acc_xs = [1.0 if p["correct"] else 0.0 for p in ps]
        acc = sum(acc_xs) / len(acc_xs)
        lo, hi = bootstrap_ci(acc_xs)
        reader = _mean([p.get("n_tokens") for p in ps])
        qids = {p["qid"] for p in ps}
        rows.append({
            "method": method, "budget": budget, "qtype": qtype, "n_questions": len(ps),
            "accuracy": acc, "ci95_low": lo, "ci95_high": hi,
            "evidence_recall": _mean([p.get("evidence_recall") for p in ps]),
            "avg_reader_input_tokens": reader,
            "construction_tokens_per_q": _mean([stage_tok[(method, q)]["construction"] for q in qids]),
            "query_time_tokens_per_q": _mean([stage_tok[(method, q)]["query_time"] for q in qids]),
            "system_usd_per_q": _mean([stage_usd[(method, q)] for q in qids]),
            "acc_per_1k_reader_tokens": acc / (reader / 1000) if reader else None,
            "run_ids": ";".join(sorted({p["run_id"] for p in ps})),
        })

    # rank methods within each (budget, qtype) by accuracy
    by_slot: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_slot[(r["budget"], r["qtype"])].append(r)
    for slot in by_slot.values():
        slot.sort(key=lambda r: -r["accuracy"])
        best = slot[0]
        for i, r in enumerate(slot):
            r["rank_at_budget"] = i + 1
            r["best_at_budget"] = best["method"]
            r["delta_vs_best"] = r["accuracy"] - best["accuracy"]

    budget_key = lambda b: (isinstance(b, str), b if isinstance(b, int) else 0, str(b))
    rows.sort(key=lambda r: (r["qtype"] != "all", r["qtype"], budget_key(r["budget"]), r["rank_at_budget"]))
    return rows


def write_csv(rows: list[dict], out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v))
                        for k, v in ((c, r.get(c)) for c in COLUMNS)})
    return out
