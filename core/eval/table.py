"""Performance tables from ONE logged run (never hand-edited).
<run_id>.csv          one row per system x budget x qtype (plus qtype=ALL), 95% bootstrap CIs
<run_id>_summary.csv  the qtype=ALL rows only: the headline system x budget comparison"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from core.eval.compare import bootstrap_ci


def stat(rs, key):
    xs = [float(r[key]) for r in rs if r.get(key) is not None]
    if not xs:
        return "", "", ""
    lo, hi = bootstrap_ci(xs)
    return round(sum(xs) / len(xs), 4), round(lo, 4), round(hi, 4)


def build_tables(run: Path, out_dir: Path) -> tuple[Path, Path]:
    preds = [json.loads(line) for line in open(run / "predictions.jsonl")]
    costs = [json.loads(line) for line in open(run / "costs.jsonl")] if (run / "costs.jsonl").exists() else []
    # System cost (judge excluded): LLM tokens by stage, embedding tokens separately.
    n_q, n_b = defaultdict(set), defaultdict(set)
    for p in preds:
        n_q[p["system"]].add(p["qid"])
        n_b[p["system"]].add(p["budget"])
    spent = defaultdict(int)
    for c in costs:
        if c.get("stage") in ("reader", "judge"):
            continue
        kind = "llm" if c.get("kind") == "chat" else "embed"
        spent[(c.get("method"), c.get("stage"), kind)] += c.get("input_tokens", 0) + c.get("output_tokens", 0)

    groups = defaultdict(list)
    for p in preds:
        groups[(p["system"], p["budget"], p["qtype"])].append(p)
        groups[(p["system"], p["budget"], "ALL")].append(p)

    rows = []
    for (s, b, qt), rs in sorted(groups.items(), key=lambda x: (x[0][1], x[0][0], x[0][2] != "ALL", x[0][2])):
        nq = len(n_q[s])
        row = {"system": s, "system_name": rs[0]["system_name"], "budget": b, "qtype": qt, "n": len(rs)}
        for key, col in (("correct", "accuracy"), ("verbatim_recall", "verbatim_recall"),
                         ("evidence_recall", "evidence_recall")):  # accuracy is empty for eval=recall runs
            row[col], row[f"{col}_ci_low"], row[f"{col}_ci_high"] = stat(rs, key)
        row["avg_context_tokens"] = round(sum(r["n_tokens"] for r in rs) / len(rs), 1)
        row["avg_view_tokens"] = round(sum(r["view_tokens"] for r in rs) / len(rs), 1)
        row["llm_build_tokens_per_q"] = spent[(s, "construction", "llm")] // nq  # once per question
        row["llm_query_tokens_per_query"] = spent[(s, "query_time", "llm")] // (nq * len(n_b[s]))  # per budget
        row["embedding_tokens_per_q"] = (spent[(s, "construction", "embed")]
                                         + spent[(s, "query_time", "embed")] // len(n_b[s])) // nq
        row["run_id"] = run.name
        rows.append(row)

    out_dir.mkdir(parents=True, exist_ok=True)
    full, summary = out_dir / f"{run.name}.csv", out_dir / f"{run.name}_summary.csv"
    for path, rs in ((full, rows), (summary, [r for r in rows if r["qtype"] == "ALL"])):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rs)
    return full, summary
