"""Compare context-optimization systems on LongMemEval-S. Usage:
  python scripts/run_compare.py                                   # configs/compare.yaml
  python scripts/run_compare.py compare.eval=qa compare.n_questions=20 "compare.systems=[bm25,eco]"
"""
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="No price")

import json  # noqa: E402

from core.config import CONFIG_DIR, REPO_ROOT, load_config  # noqa: E402
from core.data.longmemeval import load_longmemeval  # noqa: E402
from core.data.split import load_split, stratified_dev  # noqa: E402
from core.data.views import apply_view  # noqa: E402
from core.eval.table import build_tables  # noqa: E402
from core.llm.cost import PlannedCalls, Pricing, cost_guard  # noqa: E402
from core.llm.models import get_chat_model, get_chat_pool, get_embeddings  # noqa: E402
from core.llm.structured import session_batches  # noqa: E402
from core.rundir import new_run_dir  # noqa: E402
from core.runner import Runner  # noqa: E402
from systems import LLM_FREE, NEEDS_EMBEDDINGS  # noqa: E402

cfg = load_config(CONFIG_DIR / "compare.yaml", sys.argv[1:])
c = cfg["compare"]
systems, budgets, mode = c["systems"], cfg["budgets"], c["eval"]
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
split = load_split(REPO_ROOT / cfg["data"]["split_file"], data_path)
if c["split"] == "test":
    print("WARNING: running on the TEST split. Use it only for the final report; never tune on it.")
ids = set(split[c["split"]])
qs = [q for q in load_longmemeval(data_path) if q.qid in ids]
if c["n_questions"]:
    pick = set(stratified_dev(qs, c["n_questions"], c["sample_seed"]))
    qs = [q for q in qs if q.qid in pick]
if not c.get("include_abstention"):
    qs = [q for q in qs if not q.is_abstention]
qs = [apply_view(q, c["view"]) for q in qs]
empty = [q.qid for q in qs if not q.haystack]
qs = [q for q in qs if q.haystack]
print(f"{len(qs)} {c['split']} questions, view={c['view']}, eval={mode}, systems={systems}, budgets={budgets}"
      + (f" ({len(empty)} skipped: empty view)" if empty else ""))

# Load only what the run needs (LLM-free recall runs need no chat model at all).
need_chat = mode == "qa" or bool(set(systems) - LLM_FREE)
chat = get_chat_pool() if need_chat else [None]
if need_chat and c.get("pool_models"):
    chat = [m for m in chat if (getattr(m, "model_name", None) or m.model) in c["pool_models"]]
    if not chat:
        sys.exit(f"No pool model matches {c['pool_models']}")
judge_m = get_chat_model("judge") if mode == "qa" else None
emb = get_embeddings() if NEEDS_EMBEDDINGS & set(systems) else None
if emb is not None:
    try:
        emb.embed_documents(["preflight"])
    except Exception as e:
        sys.exit(f"Embeddings unavailable ({type(e).__name__}: {e}). Start Ollama and pull the embedding model.")

nq, nb, m = len(qs), len(budgets), cfg["models"]
n_batches = sum(len(session_batches(q.haystack)) for q in qs)
plan = [PlannedCalls(m["extraction"], n_batches * (("eco_facts" in systems) + ("memorycpt" in systems)), 6500, 1500, "construction"),
        PlannedCalls(m["reader"], nq * nb * (7 * ("refind" in systems) + ("memorycpt" in systems)), 4000, 300, "query_time")]
if mode == "qa":
    plan += [PlannedCalls(m["reader"], nq * nb * len(systems), 1300, 400, "reader"),
             PlannedCalls(m["judge"], nq * nb * len(systems), 250, 100, "judge")]
cost_guard(plan, Pricing(cfg.get("pricing", {})), cfg.get("max_usd"))
print(f"~{sum(p.n_calls for p in plan)} chat-model calls (cached calls are free and instant)")

run = new_run_dir(cfg, f"compare_{c['split']}_{c['view']}_{mode}")
print(f"Run dir: {run.relative_to(REPO_ROOT)}")
t0 = time.time()
Runner(cfg, run, chat, judge_m, emb, eval_mode=mode).run(qs, systems, budgets, workers=c["workers"])
print(f"Done in {(time.time() - t0) / 60:.1f} min.")

rows = [json.loads(line) for line in open(run / "predictions.jsonl")] if (run / "predictions.jsonl").exists() else []
agg = defaultdict(list)
for r in rows:
    agg[(r["system"], r["budget"])].append(r)
metric = "correct" if mode == "qa" else "verbatim_recall"
print(f"\n{'system':<12}{'B':>6}{'n':>5}{metric:>17}{'tokens':>8}")
for (s, b), rs in sorted(agg.items(), key=lambda x: (x[0][1], x[0][0])):
    v = [float(r[metric]) for r in rs if r.get(metric) is not None]
    print(f"{s:<12}{b:>6}{len(rs):>5}{(sum(v) / len(v) if v else float('nan')):>17.3f}"
          f"{sum(r['n_tokens'] for r in rs) / len(rs):>8.0f}")
if rows:
    for path in build_tables(run, REPO_ROOT / "results" / "tables"):
        print(f"wrote {path.relative_to(REPO_ROOT)}")
if (run / "errors.jsonl").exists():
    print(f"\n{sum(1 for _ in open(run / 'errors.jsonl'))} jobs failed; see {(run / 'errors.jsonl').relative_to(REPO_ROOT)}")
