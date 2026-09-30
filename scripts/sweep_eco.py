"""Tune ECO's two selection parameters on the DEV split only (no LLM calls; embeddings via Ollama, cached).
Usage: python scripts/sweep_eco.py
Grid: rel_tau in {None (flat RRF), 20, 10, 5, 2} x sim_floor in {0.0, 0.5, 0.7}, B in {250, 500, 1000, 2000}.
Selection rule, fixed before looking at results: highest mean verbatim recall over the four budgets.
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
from core.eval.recall import evidence_recall  # noqa: E402
from core.llm.cache import DiskCache  # noqa: E402
from core.llm.client import EmbeddingClient  # noqa: E402
from core.llm.cost import CostLogger, Pricing  # noqa: E402
from core.llm.models import get_embeddings  # noqa: E402
from core.rundir import new_run_dir  # noqa: E402
from systems.eco.method import ECO  # noqa: E402

TAUS = [None, 20, 10, 5, 2]
FLOORS = [0.0, 0.5, 0.7]

cfg = load_config(CONFIG_DIR / "compare.yaml", sys.argv[1:])
assert cfg["compare"]["split"] == "dev", "tuning is dev-only"
budgets = cfg["budgets"]
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
dev = set(load_split(REPO_ROOT / cfg["data"]["split_file"], data_path)["dev"])
qs = [apply_view(q, cfg["compare"]["view"]) for q in load_longmemeval(data_path)
      if q.qid in dev and not q.is_abstention]
qs = [q for q in qs if q.haystack]

run = new_run_dir(cfg, "eco_sweep_dev")
emb = EmbeddingClient(get_embeddings(), DiskCache(REPO_ROOT / cfg["paths"]["cache"]),
                      CostLogger(run / "costs.jsonl", Pricing(cfg.get("pricing", {}))), method="eco")
eco = ECO(Services(llms={}, embedder=emb))
print(f"{len(qs)} dev questions, {len(TAUS) * len(FLOORS)} settings, budgets {budgets}. Run dir: {run.relative_to(REPO_ROOT)}")

scores = {}  # (tau, floor, B) -> list of verbatim recall
with open(run / "sweep.jsonl", "w") as out:
    for n, q in enumerate(qs, 1):
        memory = eco.build_memory(q)
        for tau, floor in itertools.product(TAUS, FLOORS):
            eco.rel_tau, eco.sim_floor = tau, floor
            for b in budgets:
                res = eco.build_context(q.question, memory, b, q.question_date)
                assert res.n_tokens <= b
                vr = evidence_recall(res.verbatim_turn_ids, q.evidence_turn_ids)
                scores.setdefault((tau, floor, b), []).append(vr)
                out.write(json.dumps({"qid": q.qid, "qtype": q.qtype, "rel_tau": tau, "sim_floor": floor,
                                      "budget": b, "verbatim_recall": vr, "n_tokens": res.n_tokens}) + "\n")
        if n % 10 == 0:
            print(f"  {n}/{len(qs)} questions")

rows = []
for tau, floor in itertools.product(TAUS, FLOORS):
    row = {"rel_tau": tau if tau is not None else "flat", "sim_floor": floor}
    means = []
    for b in budgets:
        xs = scores[(tau, floor, b)]
        row[f"B{b}"] = round(sum(xs) / len(xs), 4)
        means.append(row[f"B{b}"])
    row["mean"] = round(sum(means) / len(means), 4)
    rows.append(row)
rows.sort(key=lambda r: -r["mean"])
table = REPO_ROOT / "results" / "tables" / f"{run.name}.csv"
with open(table, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"\n{'rel_tau':>8}{'floor':>7}" + "".join(f"{'B=' + str(b):>8}" for b in budgets) + f"{'mean':>8}")
for r in rows:
    print(f"{str(r['rel_tau']):>8}{r['sim_floor']:>7}" + "".join(f"{r[f'B{b}']:>8.3f}" for b in budgets) + f"{r['mean']:>8.3f}")
print(f"\nwrote {table.relative_to(REPO_ROOT)}")
