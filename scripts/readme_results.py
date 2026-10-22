"""Render the README results block from a generated QA summary table (never typed by hand). Usage:
  python scripts/readme_results.py [results/tables/<run_id>_summary.csv]   (default: latest *_qa_summary.csv)
Replaces the text between <!-- results:start --> and <!-- results:end --> in README.md.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import REPO_ROOT  # noqa: E402

SYSTEMS = [("eco_spans", "**ECO (ours)**"), ("refind", "ReFind (reimpl.)"),
           ("memorycpt", "MemoryCPT-lite (reimpl.)"), ("bm25", "BM25 (turns)"),
           ("hybrid", "Hybrid BM25+dense (turns)"), ("mmr", "MMR (turns)"),
           ("span_hybrid", "Hybrid BM25+dense (sentences)"), ("span_mmr", "MMR (sentences)"),
           ("oracle", "Oracle (upper bound)")]
START, END = "<!-- results:start -->", "<!-- results:end -->"

tables = REPO_ROOT / "results" / "tables"
src = Path(sys.argv[1]) if len(sys.argv) > 1 else max(tables.glob("*_qa_summary.csv"))
rows = list(csv.DictReader(open(src if src.is_absolute() else REPO_ROOT / src)))
by = {(r["system"], int(r["budget"])): r for r in rows}
budgets = sorted({int(r["budget"]) for r in rows})
n = int(rows[0]["n"])


def acc(r):
    k = round(float(r["accuracy"]) * n)
    return f"{k}/{n} ({float(r['accuracy_ci_low']):.2f}-{float(r['accuracy_ci_high']):.2f})"


run_id = rows[0]["run_id"]                      # e.g. 20261022-120500_compare_dev_oracle_qa
_, split, view, mode = run_id.replace("_compare", "").split("_", 3)[:4]
lines = [f"QA accuracy: {n} stratified {split} questions, `{view}` view, one shared reader and judge "
         f"(`gpt-oss-20b`), one token budget set in `configs/compare.yaml` (95% bootstrap CI). "
         f"Source: `results/tables/{src.name}`. Every row is a whole system, end to end.", "",
         "| System | " + " | ".join(f"Accuracy, B={b}" for b in budgets)
         + " | Reader context tokens (B=" + str(budgets[-1]) + ") | LLM build tokens / question | LLM query-time tokens / query |",
         "|---|" + "---|" * (len(budgets) + 3)]
for key, label in SYSTEMS:
    rs = [by[(key, b)] for b in budgets if (key, b) in by]
    if not rs:
        continue
    last = by[(key, budgets[-1])]
    lines.append(f"| {label} | " + " | ".join(acc(by[(key, b)]) for b in budgets)
                 + f" | {float(last['avg_context_tokens']):.0f} | {int(last['llm_build_tokens_per_q']):,}"
                 f" | {int(last['llm_query_tokens_per_query']):,} |")
lines += ["", "Token columns count chat-model calls only (embeddings run locally and are listed separately in the "
          "CSV); judge calls are excluded from system cost. The oracle row is the ceiling this sample allows: the "
          "reader cannot answer questions whose evidence it would get wrong even with the gold sessions first."]

readme = REPO_ROOT / "README.md"
text = readme.read_text()
i, j = text.index(START) + len(START), text.index(END)
readme.write_text(text[:i] + "\n" + "\n".join(lines) + "\n" + text[j:])
print(f"README results block updated from {src.name}")

# ---- Recall block: the headline test-split comparison (between <!-- recall:start --> and <!-- recall:end -->) ----
RSTART, REND = "<!-- recall:start -->", "<!-- recall:end -->"
LABELS = {"eco": "**ECO (ours)**", "hybrid": "Hybrid (BM25 + dense)", "bm25": "BM25", "mmr": "MMR", "dense": "Dense",
          "recent": "Most recent", "oracle": "Oracle (upper bound)"}
summ = max(tables.glob("*_compare_test_oracle_recall_summary.csv"))
run_id = summ.name.replace("_summary.csv", "")
srows = list(csv.DictReader(open(summ)))
paired = list(csv.DictReader(open(tables / f"{run_id}_paired_eco.csv")))
bytype = list(csv.DictReader(open(tables / f"{run_id}.csv")))
rb = sorted({int(r["budget"]) for r in srows})
nq = srows[0]["n"]
cell = {(r["system"], int(r["budget"])): r for r in srows}
best = {b: max((s for s in ("bm25", "hybrid", "mmr", "dense")), key=lambda s: float(cell[(s, b)]["verbatim_recall"]))
        for b in rb}

rl = [f"Test split, run once after ECO's settings were frozen on dev: {nq} answerable questions, oracle view. Metric: "
      "share of gold evidence turns whose raw text reaches the reader (verbatim recall), with 95% bootstrap CIs. "
      f"Source: `results/tables/{summ.name}`.", "",
      "| System | " + " | ".join(f"B={b}" for b in rb) + " |", "|---|" + "---|" * len(rb)]
for s in ("eco", "hybrid", "bm25", "mmr", "dense", "recent", "oracle"):
    cells = []
    for b in rb:
        r = cell[(s, b)]
        v = f"{float(r['verbatim_recall']):.3f}"
        v = f"**{v}**" if s != "oracle" and float(r["verbatim_recall"]) == max(
            float(cell[(x, b)]["verbatim_recall"]) for x in ("eco", "hybrid", "bm25", "mmr", "dense", "recent")) else v
        cells.append(f"{v} ({float(r['verbatim_recall_ci_low']):.2f}-{float(r['verbatim_recall_ci_high']):.2f})")
    rl.append(f"| {LABELS[s]} | " + " | ".join(cells) + " |")

rl += ["", "ECO minus the strongest baseline at each budget, paired over the same questions (95% paired bootstrap CI). "
       f"Source: `results/tables/{run_id}_paired_eco.csv`.", "",
       "| Budget | Strongest baseline | Difference | 95% CI | ECO better / worse (questions) |", "|---|---|---|---|---|"]
for b in rb:
    p = next(p for p in paired if int(p["budget"]) == b and p["baseline"] == best[b])
    rl.append(f"| {b} | {LABELS[best[b]]} | {float(p['mean_diff']):+.3f} | [{float(p['ci_low']):+.3f}, "
              f"{float(p['ci_high']):+.3f}] | {p['better']} / {p['worse']} |")

qts = sorted({r["qtype"] for r in bytype if r["qtype"] != "ALL"})
bt = {(r["system"], int(r["budget"]), r["qtype"]): r for r in bytype}
rl += ["", "ECO minus the strongest baseline by question type (verbatim recall).", "",
       "| Question type | n | " + " | ".join(f"B={b}" for b in rb) + " |", "|---|---|" + "---|" * len(rb)]
for qt in qts:
    diffs = []
    for b in rb:
        top = max(float(bt[(s, b, qt)]["verbatim_recall"]) for s in ("bm25", "hybrid", "mmr", "dense"))
        diffs.append(f"{float(bt[('eco', b, qt)]['verbatim_recall']) - top:+.3f}")
    rl.append(f"| {qt} | {bt[('eco', rb[0], qt)]['n']} | " + " | ".join(diffs) + " |")

text = readme.read_text()
i, j = text.index(RSTART) + len(RSTART), text.index(REND)
readme.write_text(text[:i] + "\n" + "\n".join(rl) + "\n" + text[j:])
print(f"README recall block updated from {summ.name}")
