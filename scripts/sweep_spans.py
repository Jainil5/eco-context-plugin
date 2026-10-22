"""Tune ECO-spans' two selection parameters on the DEV split only (no LLM calls; embeddings via Ollama, cached).

Turn-level ECO's tau/floor were swept in scripts/sweep_eco.py; those values were inherited by the span-level system
unchanged, which is only fair to the method if the span granularity is tuned on its own terms. Same grid shape, same
dev-only rule, but the objective has to change: at sentence granularity verbatim turn recall is zero by design, so the
selection rule is fixed in advance as highest mean content_recall over the four budgets (gold sentences that reach the
reader), with answer_in_ctx reported next to it and used only to break ties.

Grid: rel_tau in {None (raw fused score), 20, 10, 5, 2, 1} x sim_floor in {0.0, 0.5, 0.7, 0.85},
B in the budgets in configs/compare.yaml.
Writes results/runs/<run_id>/sweep.jsonl and results/tables/<run_id>.csv (one row per setting).
"""
import csv
import itertools
import json
import sys
import warnings
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

TAUS = [None, 20, 10, 5, 2, 1]
FLOORS = [0.0, 0.5, 0.7, 0.85]

cfg = load_config(CONFIG_DIR / "compare.yaml", sys.argv[1:])
assert cfg["compare"]["split"] == "dev", "tuning is dev-only"
budgets = cfg["budgets"]
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
dev = set(load_split(REPO_ROOT / cfg["data"]["split_file"], data_path)["dev"])
qs = [apply_view(q, cfg["compare"]["view"]) for q in load_longmemeval(data_path)
      if q.qid in dev and not q.is_abstention]
qs = [q for q in qs if q.haystack]

run = new_run_dir(cfg, "eco_sweep_spans_dev")
emb = EmbeddingClient(get_embeddings(), DiskCache(REPO_ROOT / cfg["paths"]["cache"]),
                      CostLogger(run / "costs.jsonl", Pricing(cfg.get("pricing", {}))), method="eco_spans")
eco = ECOspans(Services(llms={}, embedder=emb))
print(f"{len(qs)} dev questions, {len(TAUS) * len(FLOORS)} settings, budgets {budgets}. Run dir: {run.relative_to(REPO_ROOT)}")

cr: dict[tuple, list[float]] = {}
ac: dict[tuple, list[float]] = {}
with open(run / "sweep.jsonl", "w") as out:
    for n, q in enumerate(qs, 1):
        memory = eco.build_memory(q)
        gold = set(q.evidence_turn_ids)
        gold_texts = [t.text for t in q.haystack if t.turn_id in gold]
        for tau, floor in itertools.product(TAUS, FLOORS):
            eco.rel_tau, eco.sim_floor = tau, floor
            for b in budgets:
                res = eco.build_context(q.question, memory, b, q.question_date)
                assert res.n_tokens <= b
                c, a = content_recall(res.context, gold_texts), answer_in_context(res.context, q.answer)
                cr.setdefault((tau, floor, b), []).append(c)
                ac.setdefault((tau, floor, b), []).append(a)
                out.write(json.dumps({"qid": q.qid, "qtype": q.qtype, "rel_tau": tau, "sim_floor": floor,
                                      "budget": b, "content_recall": c, "answer_in_ctx": a,
                                      "n_tokens": res.n_tokens}) + "\n")
        if n % 10 == 0:
            print(f"  {n}/{len(qs)} questions")


def mean(xs: list[float | None]) -> float:
    v = [x for x in xs if x is not None]
    return sum(v) / len(v) if v else float("nan")


rows = []
for tau, floor in itertools.product(TAUS, FLOORS):
    row = {"rel_tau": tau if tau is not None else "fused", "sim_floor": floor}
    per = [mean(cr[(tau, floor, b)]) for b in budgets]
    for b, v in zip(budgets, per):
        row[f"B{b}"] = round(v, 4)
    row["mean_content"] = round(sum(per) / len(per), 4)
    row["mean_answer_in_ctx"] = round(sum(mean(ac[(tau, floor, b)]) for b in budgets) / len(budgets), 4)
    rows.append(row)
rows.sort(key=lambda r: (-r["mean_content"], -r["mean_answer_in_ctx"]))

table = REPO_ROOT / "results" / "tables" / f"{run.name}.csv"
with table.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

print(f"\n{'rel_tau':>8}{'floor':>7}" + "".join(f"{'B=' + str(b):>8}" for b in budgets)
      + f"{'mean_cr':>9}{'ans_c':>7}")
for r in rows:
    print(f"{str(r['rel_tau']):>8}{r['sim_floor']:>7}" + "".join(f"{r[f'B{b}']:>8.3f}" for b in budgets)
          + f"{r['mean_content']:>9.3f}{r['mean_answer_in_ctx']:>7.3f}")
print(f"\nwinner: rel_tau={rows[0]['rel_tau']} sim_floor={rows[0]['sim_floor']} "
      f"mean content_recall={rows[0]['mean_content']}")
print(f"wrote {table.relative_to(REPO_ROOT)}")
