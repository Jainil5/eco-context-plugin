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

SYSTEMS = [("eco", "ECO (ours)"), ("memorycpt", "MemoryCPT-lite (reimpl.)"), ("refind", "ReFind (reimpl.)")]
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


lines = [f"QA accuracy on {n} stratified dev questions, oracle view, reader and judge `gpt-oss-20b` "
         f"(95% bootstrap CI). Source: `results/tables/{src.name}`.", "",
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
if ("oracle", budgets[-1]) in by:
    lines += ["", "Reference, same run: the oracle (all gold evidence turns first) scores "
              + ", ".join(f"{acc(by[('oracle', b)]).split(' ')[0]} at B={b}" for b in budgets)
              + ", so the reader caps accuracy on this sample. Token columns count LLM calls only (embeddings "
              "run locally and are listed separately in the CSV); judge calls are excluded."]

readme = REPO_ROOT / "README.md"
text = readme.read_text()
i, j = text.index(START) + len(START), text.index(END)
readme.write_text(text[:i] + "\n" + "\n".join(lines) + "\n" + text[j:])
print(f"README results block updated from {src.name}")
