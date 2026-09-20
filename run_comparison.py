"""Compare normal agent (full history) vs ECO context optimizer; write a verification CSV."""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from baselines import BASELINE_AGENT, BUILDERS, ECO_AGENT
from data import load_samples_dir
from qa import generate_answer, judge_answer
from tokens import count_tokens

from run_baselines import full_history_tokens, load_examples

DEFAULT_BUDGETS = [500, 1000, 2000, 4000]


def run_row(
    ex: dict,
    budget: int,
    *,
    dry_run: bool,
    judge: bool,
) -> dict:
    baseline_builder = BUILDERS[BASELINE_AGENT]
    eco_builder = BUILDERS[ECO_AGENT]
    full_tok = full_history_tokens(ex)

    t0 = time.perf_counter()
    baseline_ctx = baseline_builder(ex, budget=budget)
    baseline_ctx_tokens = count_tokens(baseline_ctx)
    if dry_run:
        baseline_answer = ""
        baseline_correct = None
        baseline_latency = time.perf_counter() - t0
    else:
        baseline_answer = generate_answer(ex, baseline_ctx)
        baseline_latency = time.perf_counter() - t0
        baseline_correct = judge_answer(ex, baseline_answer) if judge else None

    t0 = time.perf_counter()
    eco_ctx = eco_builder(ex, budget=budget)
    eco_ctx_tokens = count_tokens(eco_ctx)
    if dry_run:
        eco_answer = ""
        eco_correct = None
        eco_latency = time.perf_counter() - t0
    else:
        eco_answer = generate_answer(ex, eco_ctx)
        eco_latency = time.perf_counter() - t0
        eco_correct = judge_answer(ex, eco_answer) if judge else None

    reduction = 1 - (eco_ctx_tokens / full_tok) if full_tok else 0.0
    baseline_reduction = 1 - (baseline_ctx_tokens / full_tok) if full_tok else 0.0

    return {
        "example_id": ex["question_id"],
        "question_type": ex["question_type"],
        "question": ex["question"],
        "ground_truth": ex["answer"],
        "budget": budget,
        "full_history_tokens": full_tok,
        "baseline_strategy": BASELINE_AGENT,
        "eco_strategy": ECO_AGENT,
        "baseline_context_tokens": baseline_ctx_tokens,
        "eco_context_tokens": eco_ctx_tokens,
        "baseline_token_reduction_vs_full": round(baseline_reduction, 4),
        "eco_token_reduction_vs_full": round(reduction, 4),
        "eco_saves_tokens_vs_baseline": baseline_ctx_tokens - eco_ctx_tokens,
        "baseline_answer": baseline_answer,
        "eco_answer": eco_answer,
        "baseline_correct": baseline_correct,
        "eco_correct": eco_correct,
        "eco_wins": (
            eco_correct is True and baseline_correct is not True
            if judge and not dry_run
            else None
        ),
        "baseline_wins": (
            baseline_correct is True and eco_correct is not True
            if judge and not dry_run
            else None
        ),
        "both_correct": (
            baseline_correct is True and eco_correct is True
            if judge and not dry_run
            else None
        ),
        "baseline_latency_sec": round(baseline_latency, 3),
        "eco_latency_sec": round(eco_latency, 3),
    }


def write_csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], judge: bool) -> dict:
    n = len(rows)
    if n == 0:
        return {}
    out = {
        "runs": n,
        "avg_baseline_context_tokens": sum(r["baseline_context_tokens"] for r in rows) / n,
        "avg_eco_context_tokens": sum(r["eco_context_tokens"] for r in rows) / n,
        "avg_eco_token_reduction_vs_full": sum(
            r["eco_token_reduction_vs_full"] for r in rows
        )
        / n,
    }
    if judge and rows[0].get("baseline_correct") is not None:
        b_ok = sum(1 for r in rows if r["baseline_correct"])
        e_ok = sum(1 for r in rows if r["eco_correct"])
        out["baseline_accuracy"] = b_ok / n
        out["eco_accuracy"] = e_ok / n
        out["eco_only_correct"] = sum(1 for r in rows if r["eco_wins"])
        out["baseline_only_correct"] = sum(1 for r in rows if r["baseline_wins"])
        out["both_correct"] = sum(1 for r in rows if r["both_correct"])
    return out


def load_comparison_examples(args) -> list[dict]:
    if getattr(args, "samples_dir", None):
        return load_samples_dir(args.samples_dir)
    examples = load_examples(args.split)
    if args.limit:
        examples = examples[: args.limit]
    return examples


def run(args):
    examples = load_comparison_examples(args)
    budgets = args.budgets
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"comparison_{stamp}.csv"
    jsonl_path = out_dir / f"comparison_{stamp}.jsonl"
    rows = []

    for ex in examples:
        for budget in budgets:
            row = run_row(
                ex,
                budget,
                dry_run=args.dry_run,
                judge=args.judge,
            )
            rows.append(row)
            with open(jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
            print(
                f"budget={budget} id={ex['question_id'][:8]} "
                f"base={row['baseline_context_tokens']} eco={row['eco_context_tokens']} "
                f"ok base={row['baseline_correct']} eco={row['eco_correct']}"
            )

    write_csv(rows, csv_path)
    summary = summarize(rows, args.judge and not args.dry_run)
    summary_path = out_dir / f"comparison_{stamp}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"csv: {csv_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="CSV comparison: normal agent (full_history) vs ECO optimizer"
    )
    p.add_argument(
        "--samples-dir",
        default="samples",
        help="folder with conversation_*.json (default: samples/)",
    )
    p.add_argument(
        "--split",
        default="sample",
        help="if samples-dir missing, use dataset split",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    samples_root = Path(args.samples_dir)
    if not samples_root.is_dir() or not list(samples_root.glob("conversation_*.json")):
        print(f"missing conversations in {samples_root}/; run: python export_samples.py")
        args.samples_dir = None
    run(args)
