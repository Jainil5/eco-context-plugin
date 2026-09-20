import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from baselines import BUILDERS, METHOD_NAMES
from data import load_split
from qa import generate_answer, judge_answer
from tokens import count_tokens

DEFAULT_BUDGETS = [500, 1000, 2000, 4000, 8000]


def load_examples(split: str) -> list[dict]:
    if split == "sample":
        path = Path("dataset/longmemeval_sample.json")
        with open(path) as f:
            return json.load(f)
    return load_split(split)


def full_history_tokens(entry: dict) -> int:
    from context_format import format_sessions

    text = format_sessions(entry["haystack_dates"], entry["haystack_sessions"])
    return count_tokens(text)


def run(args):
    examples = load_examples(args.split)
    if args.limit:
        examples = examples[: args.limit]
    methods = METHOD_NAMES if "all" in args.methods else args.methods
    budgets = args.budgets
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = out_dir / f"baseline_{args.split}_{stamp}.jsonl"
    rows = []

    for ex in examples:
        full_tok = full_history_tokens(ex)
        for method in methods:
            if method not in BUILDERS:
                raise ValueError(f"unknown method {method}")
            builder = BUILDERS[method]
            for budget in budgets:
                t0 = time.perf_counter()
                context = builder(ex, budget=budget)
                ctx_tokens = count_tokens(context)
                if args.dry_run:
                    answer = ""
                    correct = None
                    latency = time.perf_counter() - t0
                else:
                    answer = generate_answer(ex, context)
                    latency = time.perf_counter() - t0
                    correct = judge_answer(ex, answer) if args.judge else None
                row = {
                    "method": method,
                    "example_id": ex["question_id"],
                    "question_type": ex["question_type"],
                    "budget": budget,
                    "context_tokens": ctx_tokens,
                    "full_history_tokens": full_tok,
                    "token_reduction": 1 - (ctx_tokens / full_tok) if full_tok else 0,
                    "answer": answer,
                    "ground_truth": ex["answer"],
                    "correct": correct,
                    "latency_sec": round(latency, 3),
                    "cost": None,
                }
                rows.append(row)
                with open(results_path, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(
                    f"{method} budget={budget} id={ex['question_id'][:8]} "
                    f"ctx={ctx_tokens} ok={correct}"
                )

    summary_path = out_dir / f"baseline_{args.split}_{stamp}_summary.json"
    by_method = {}
    for r in rows:
        m = r["method"]
        by_method.setdefault(m, {"n": 0, "correct": 0, "ctx_tokens": []})
        by_method[m]["n"] += 1
        if r["correct"]:
            by_method[m]["correct"] += 1
        by_method[m]["ctx_tokens"].append(r["context_tokens"])
    summary = {}
    for m, v in by_method.items():
        acc = v["correct"] / v["n"] if v["n"] and args.judge else None
        summary[m] = {
            "accuracy": acc,
            "avg_context_tokens": sum(v["ctx_tokens"]) / len(v["ctx_tokens"]),
            "runs": v["n"],
        }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"results: {results_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="sample", help="s | m | oracle | sample")
    p.add_argument("--methods", nargs="+", default=["all"])
    p.add_argument("--budgets", nargs="+", type=int, default=DEFAULT_BUDGETS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--judge", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="build context only, no LLM QA")
    args = p.parse_args()
    run(args)
