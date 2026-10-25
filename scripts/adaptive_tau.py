"""Adaptive-tau experiment, DEV split only (no LLM calls; embeddings via Ollama, cached).

The frozen rel_tau=2 was chosen on the oracle view, where the pool is tiny; the hypothesis is that a per-question
tau estimated from the candidates' dense-score decay travels to pools the retriever does not separate cleanly.
Two rules (systems/eco/objective.py): fit = exponential decay rate of the top-20 dense-score excess over the pool
median; elbow = how many candidates sit above the midpoint of best-vs-median. Both are compared against the frozen tau=2 and against fixed
taus that a global re-sweep could also have chosen — adaptive only counts if it beats the best FIXED tau, not just
the frozen one.

Same question set, budgets and floor (0.7) as scripts/sweep_spans.py; metrics are the two span-level proxies
(content_recall, answer_in_ctx). Writes results/runs/<run_id>/sweep.jsonl and results/tables/<run_id>.csv,
plus a paired-bootstrap table vs the frozen setting (5000 resamples, seed 0, same convention as paired_compare).
"""
import csv
import json
import random
import sys
import warnings
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore", message="No price")

from core.config import CONFIG_DIR, REPO_ROOT, load_config  # noqa: E402
from core.context import Services  # noqa: E402
from core.data.longmemeval import load_longmemeval  # noqa: E402
from core.data.split import load_split  # noqa: E402
from core.data.views import apply_view  # noqa: E402
from core.eval.recall import answer_in_context, content_recall  # noqa: E402
from core.llm.cache import DiskCache  # noqa: E402
from core.llm.client import EmbeddingClient  # noqa: E402
from core.llm.cost import CostLogger, Pricing  # noqa: E402
from core.llm.models import get_embeddings  # noqa: E402
from core.rundir import new_run_dir  # noqa: E402
from systems.eco.spans import ECOspans  # noqa: E402

FROZEN = "tau2_frozen"
SETTINGS = {  # name -> (rel_tau, auto_tau); sim_floor is the frozen 0.7 everywhere
    FROZEN: (2.0, None),
    "tau5": (5.0, None),
    "tau10": (10.0, None),
    "tau20": (20.0, None),
    "auto_fit": (2.0, "fit"),
    "auto_elbow": (2.0, "elbow"),
}
METRICS = ("content_recall", "answer_in_ctx")

cfg = load_config(CONFIG_DIR / "compare.yaml", sys.argv[1:])
assert cfg["compare"]["split"] == "dev", "tuning is dev-only"
assert cfg["compare"]["view"] == "oracle", "this experiment assumes the oracle view (same set as sweep_spans)"
budgets = cfg["budgets"]
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
dev = set(load_split(REPO_ROOT / cfg["data"]["split_file"], data_path)["dev"])
qs = [apply_view(q, cfg["compare"]["view"]) for q in load_longmemeval(data_path)
      if q.qid in dev and not q.is_abstention]
qs = [q for q in qs if q.haystack]
if cfg["compare"]["n_questions"]:  # smoke runs only; the recorded experiment uses the whole dev split
    qs = qs[: int(cfg["compare"]["n_questions"])]

run = new_run_dir(cfg, "adaptive_tau_dev")
emb = EmbeddingClient(get_embeddings(), DiskCache(REPO_ROOT / cfg["paths"]["cache"]),
                      CostLogger(run / "costs.jsonl", Pricing(cfg.get("pricing", {}))), method="adaptive_tau")
eco = ECOspans(Services(llms={}, embedder=emb))
print(f"{len(qs)} dev questions, {len(SETTINGS)} settings, budgets {budgets}. Run dir: {run.relative_to(REPO_ROOT)}")

vals = defaultdict(list)  # (setting, budget, metric) -> per-question values (None kept out)
taus = defaultdict(list)  # setting -> per-question tau_used
with open(run / "sweep.jsonl", "w", encoding="utf-8") as out:
    for n, q in enumerate(qs, 1):
        memory = eco.build_memory(q)
        gold_texts = [t.text for t in q.haystack if t.turn_id in set(q.evidence_turn_ids)]
        for name, (tau, auto) in SETTINGS.items():
            eco.rel_tau, eco.auto_tau = tau, auto
            for b in budgets:
                res = eco.build_context(q.question, memory, b, q.question_date)
                assert res.n_tokens <= b
                c, a = content_recall(res.context, gold_texts), answer_in_context(res.context, q.answer)
                t = res.trace.get("tau_used")
                for metric, v in ((METRICS[0], c), (METRICS[1], a)):
                    if v is not None:
                        vals[(name, b, metric)].append((q.qid, float(v)))
                if t is not None:
                    taus[name].append(t)
                out.write(json.dumps({"qid": q.qid, "qtype": q.qtype, "setting": name, "budget": b,
                                      "content_recall": c, "answer_in_ctx": a, "n_tokens": res.n_tokens,
                                      "tau_used": t}) + "\n")
        if n % 10 == 0:
            print(f"  {n}/{len(qs)} questions")
    out.flush()


def mean(pairs):
    return sum(v for _, v in pairs) / len(pairs) if pairs else float("nan")


def paired_boot(a, b):
    """95% bootstrap CI of mean(a-b) over shared qids, seed 0, 5000 resamples (paired_compare's convention)."""
    da, db = dict(a), dict(b)
    qids = sorted(set(da) & set(db))
    diff = [da[q] - db[q] for q in qids]
    rng = random.Random(0)
    boots = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(5000))
    return {"n": len(diff), "mean_diff": round(sum(diff) / len(diff), 4),
            "ci_low": round(boots[125], 4), "ci_high": round(boots[4874], 4),
            "better": sum(d > 0 for d in diff), "worse": sum(d < 0 for d in diff)}


rows = []
for name in SETTINGS:
    for b in budgets:
        row = {"setting": name, "budget": b}
        for m in METRICS:
            row[m] = round(mean(vals[(name, b, m)]), 4)
        row["tau_p10"], row["tau_med"], row["tau_p90"] = (None,) * 3
        if taus[name]:
            s = sorted(taus[name])
            row["tau_p10"] = round(s[len(s) // 10], 2)
            row["tau_med"] = round(s[len(s) // 2], 2)
            row["tau_p90"] = round(s[-max(1, len(s) // 10)], 2)
        rows.append(row)

paired_rows = []
for name in SETTINGS:
    if name == FROZEN:
        continue
    for b in budgets:
        for m in METRICS:
            paired_rows.append({"setting": name, "baseline": FROZEN, "budget": b, "metric": m,
                                **paired_boot(vals[(name, b, m)], vals[(FROZEN, b, m)])})

table = REPO_ROOT / "results" / "tables" / f"{run.name}.csv"
with table.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
ptable = REPO_ROOT / "results" / "tables" / f"{run.name}_paired.csv"
with ptable.open("w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(paired_rows[0]))
    w.writeheader()
    w.writerows(paired_rows)

print(f"\n{'setting':>12}{'B':>6}{'content':>9}{'answer':>8}{'tau p10/med/p90':>18}")
for r in rows:
    print(f"{r['setting']:>12}{r['budget']:>6}{r['content_recall']:>9.3f}{r['answer_in_ctx']:>8.3f}"
          f"  {r['tau_p10']} / {r['tau_med']} / {r['tau_p90']}")
print(f"\npaired vs {FROZEN} (95% bootstrap CI):")
for r in paired_rows:
    sig = " *" if r["ci_low"] > 0 or r["ci_high"] < 0 else ""
    print(f"{r['setting']:>12} {r['metric']:>15} B={r['budget']:>5} "
          f"{r['mean_diff']:+.4f} [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] {r['better']}+/{r['worse']}- {sig}")
print(f"\nwrote {table.relative_to(REPO_ROOT)} and {ptable.relative_to(REPO_ROOT)}")
