"""Regenerate the README charts from logged result tables (nothing plotted is typed by hand). Usage:
  python scripts/make_figures.py
Writes results/figures/*.png. Every caption in the README names the run id this script reads, so re-running
after a new canonical pass reproduces the charts exactly.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

TABLES = REPO_ROOT / "results" / "tables"
OUT = REPO_ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

QA_RUN = "20261025-164642_compare_dev_oracle_qa"
PAIR_RUN = f"{QA_RUN}_paired_eco_spans_correct"
DERIVE_RUN = "20261024-023543_compare_dev_oracle_recall_paired_eco_derive_trace_metrics_answer_in_ctx"

LABEL = {"eco_spans": "ECO (ours, zero-LLM)", "memorycpt": "MemoryCPT (LLM build+query)",
         "eco_facts": "eco_facts (LLM build)", "refind": "ReFind (LLM agent)",
         "bm25": "BM25 (zero-LLM)", "span_hybrid": "span hybrid (zero-LLM)", "span_mmr": "span MMR (zero-LLM)",
         "eco": "ECO turns (zero-LLM)", "hybrid": "hybrid turns", "mmr": "MMR turns"}
LLM = {"memorycpt", "eco_facts", "refind"}
COLOR = {"eco_spans": "#1f77b4", "memorycpt": "#d62728", "eco_facts": "#e377c2", "refind": "#8c564b",
         "bm25": "#2ca02c", "span_hybrid": "#9467bd", "span_mmr": "#7f7f7f"}


def read(name):
    return list(csv.DictReader(open(TABLES / f"{name}.csv", encoding="utf-8")))


def fig_accuracy_vs_budget(rows):
    by = defaultdict(dict)
    for r in rows:
        if r["qtype"] == "ALL":
            by[r["system"]][int(r["budget"])] = r
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in sorted(by, key=lambda s: -float(by[s][500]["accuracy"])):
        bs = sorted(by[s])
        y = [float(by[s][b]["accuracy"]) for b in bs]
        lo = [float(by[s][b]["accuracy"]) - float(by[s][b]["accuracy_ci_low"]) for b in bs]
        hi = [float(by[s][b]["accuracy_ci_high"]) - float(by[s][b]["accuracy"]) for b in bs]
        ax.errorbar(bs, y, yerr=[lo, hi], marker="o", capsize=3, linewidth=2 if s == "eco_spans" else 1.4,
                    color=COLOR.get(s, "#333333"), label=LABEL.get(s, s),
                    zorder=3 if s in ("eco_spans", "memorycpt") else 2)
    ax.set_xscale("log")
    ax.set_xticks([250, 500, 1000], ["B=250", "B=500", "B=1000"])
    ax.set_xlabel("context token budget (log scale)")
    ax.set_ylabel("QA accuracy (reader + judge, 95% CI)")
    ax.set_title("Architecture claim, n=94 dev-oracle (Tier-2)\naccuracy vs context budget, 95% CI, all systems")
    ax.legend(fontsize=7.5, loc="center left")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_budget.png", dpi=150)
    plt.close(fig)


def fig_accuracy_vs_cost(rows):
    offsets = {"memorycpt": (8, 0), "eco_facts": (8, 4), "refind": (8, 0), "eco_spans": (8, 8),
               "span_hybrid": (8, -10), "bm25": (-10, 12), "span_mmr": (8, -14)}
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for r in rows:
        if r["qtype"] != "ALL" or int(r["budget"]) != 500:
            continue
        x, y = float(r["avg_context_tokens"]), float(r["accuracy"])
        ax.scatter(x, y, s=90, color=COLOR.get(r["system"], "#333333"),
                   marker="o" if r["system"] in LLM else "^", zorder=3, edgecolor="white")
        ax.annotate(LABEL.get(r["system"], r["system"]), (x, y), textcoords="offset points",
                    xytext=offsets.get(r["system"], (8, 0)), fontsize=8,
                    ha="right" if r["system"] == "bm25" else "left")
    ax.set_xlim(60, 700)
    ax.set_ylim(0.18, 0.9)
    ax.set_xlabel("mean reader context tokens (B=500)")
    ax.set_ylabel("QA accuracy (B=500)")
    ax.set_title("What the reader gets vs what it answers (B=500, n=94)\ncircle = spends LLM calls, triangle = zero-LLM")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "accuracy_vs_cost.png", dpi=150)
    plt.close(fig)


def fig_forest(rows):
    budgets = sorted({int(r["budget"]) for r in rows})
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    y = 0
    ticks, labels = [], []
    for b in budgets:
        rs = sorted((r for r in rows if int(r["budget"]) == b), key=lambda r: float(r["mean_diff"]))
        for r in rs:
            lo, d, hi = float(r["ci_low"]), float(r["mean_diff"]), float(r["ci_high"])
            sig = hi < 0 or lo > 0
            ax.plot([lo, hi], [y, y], linewidth=2, color="#d62728" if d < 0 else "#1f77b4",
                    alpha=1 if sig else 0.45)
            ax.scatter([d], [y], color="#d62728" if d < 0 else "#1f77b4",
                       marker="s" if sig else "o", zorder=3, s=45)
            ticks.append(y)
            labels.append(f"vs {r['baseline']}  (B={b})")
            y += 1
        y += 0.7
    ax.axvline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_yticks(ticks, labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("paired accuracy diff, eco_spans − baseline\n(95% paired bootstrap CI; square = excludes 0)",
                  fontsize=9)
    ax.set_title("Pre-registered claim scored (n=94 dev-oracle)\nPRIMARY fails to memorycpt at B=250; "
                 "every cell vs refind is ours", fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "paired_forest.png", dpi=150)
    plt.close(fig)


def fig_derive(rows):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    baselines = sorted({r["baseline"] for r in rows})
    width, gap = 0.24, 0.05
    xs = range(len(baselines))
    for i, b in enumerate(sorted({int(r["budget"]) for r in rows})):
        d, lo, hi = [], [], []
        for base in baselines:
            r = next(r for r in rows if int(r["budget"]) == b and r["baseline"] == base)
            d.append(float(r["mean_diff"]))
            lo.append(float(r["mean_diff"]) - float(r["ci_low"]))
            hi.append(float(r["ci_high"]) - float(r["mean_diff"]))
        pos = [x + (i - 1) * (width + gap) for x in xs]
        ax.bar(pos, d, width, yerr=[lo, hi], capsize=2.5, label=f"B={b}",
               color=["#1f77b4", "#ff7f0e", "#2ca02c"][i])
    ax.axhline(0, color="#333333", linewidth=1, linestyle="--")
    ax.set_xticks(list(xs), baselines)
    ax.set_ylabel("paired answer_in_ctx difference, eco_derive − control")
    ax.set_title("Derived facts (zero-LLM synthesis): a loss at B=250, noise above — the second null\n"
                 "eco_derive vs controls, answer_in_ctx, n=94 dev-oracle")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "eco_derive_null.png", dpi=150)
    plt.close(fig)


fig_accuracy_vs_budget(read(QA_RUN))
fig_accuracy_vs_cost(read(QA_RUN))
fig_forest(read(PAIR_RUN))
fig_derive(read(DERIVE_RUN))
for f in sorted(OUT.glob("*.png")):
    print(f"wrote {f.relative_to(REPO_ROOT)}")
