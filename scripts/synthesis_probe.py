"""Synthesis-only Tier-1 probe: does a design add answer coverage that NO single span could deliver?

On the oracle view every evidence turn fits in the largest budget, so greedy top-k already achieves the
ceiling of `answer_in_ctx` - a selection design can only tie or lose there. This script restricts the
comparison to the questions where that ceiling is not automatic: the gold answer string (and its full
content-word set) appears in NO single sentence of the view. On those, an answer can only reach the
reader if something *derived* (a count, a recency, an aggregation) puts it there - the only honest
Tier-1 headroom for eco_derive's premise.

Reads an existing logged run + its trace_metrics.jsonl (scripts/score_traces.py output); no new jobs,
no LLM calls. Classification reuses answer_in_context's matching rules, at two granularities:
  sentence-level: no single span renders the answer  -> selection (not synthesis) cannot deliver it
  turn-level:     no whole turn renders the answer   -> even a turn-granularity baseline needs synthesis
A question is "synthesis-required" only if it fails both.

Usage:
  python scripts/synthesis_probe.py results/runs/<run_id> [system=eco_derive]
Writes:
  results/tables/<run_id>_synthesis_probe.csv
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import CONFIG_DIR, REPO_ROOT, load_config  # noqa: E402
from core.context import span_units  # noqa: E402
from core.data.longmemeval import load_longmemeval  # noqa: E402
from core.data.split import load_split  # noqa: E402
from core.data.views import apply_view  # noqa: E402
from core.eval.recall import _STOP, _TOKEN  # noqa: E402
from core.llm.tokenizer import get_tokenizer  # noqa: E402
from core.text import normalize_ws, sentences  # noqa: E402

run = Path(sys.argv[1] if len(sys.argv) > 1 else "results/runs/20261024-023543_compare_dev_oracle_recall")
run = run if run.is_absolute() else REPO_ROOT / run
system = sys.argv[2] if len(sys.argv) > 2 else "eco_derive"
BASELINES = [b for b in ("eco_spans", "span_hybrid", "span_mmr") if b != system]

cfg = load_config(CONFIG_DIR / "compare.yaml")
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
split = load_split(REPO_ROOT / cfg["data"]["split_file"], data_path)
ids = set(split["dev"])
qs = [apply_view(q, "oracle") for q in load_longmemeval(data_path) if q.qid in ids and not q.is_abstention]
tok = get_tokenizer()


def reachable(text: str, answer: str, toks: set[str]) -> bool:
    """The same two tests answer_in_context applies to a whole context, here applied to one piece."""
    t = normalize_ws(text).lower()
    if normalize_ws(answer).lower() in t:
        return True
    return bool(toks) and all(x in t for x in toks)


def classify(q) -> str:
    s = str(q.answer).strip().lower()
    toks = {t for t in _TOKEN.findall(s) if t not in _STOP}
    if any(reachable(t.text, s, toks) for t in q.haystack):
        return "turn"          # a whole-turn renderer already delivers the answer
    spans = span_units(q.haystack, tok)
    if any(reachable(sp.text, s, toks) for sp in spans):
        return "span"          # some single span delivers it: selection headroom, not synthesis
    return "synthesis"


label = {q.qid: classify(q) for q in qs if q.answer is not None}
vals = defaultdict(dict)  # (system, budget) -> {qid: answer_in_ctx}
for line in open(run / "trace_metrics.jsonl", encoding="utf-8"):
    r = json.loads(line)
    if r.get("answer_in_ctx") is not None and r["qid"] in label:
        vals[(r["system"], r["budget"])][r["qid"]] = float(r["answer_in_ctx"])
budgets = sorted({b for _, b in vals})
systems_present = sorted({s for s, _ in vals})
n_synth = sum(v == "synthesis" for v in label.values())
print(f"{run.name}: {len(label)} dev-oracle questions classified: "
      + ", ".join(f"{k}={sum(1 for v in label.values() if v == k)}" for k in ("turn", "span", "synthesis")))

rows = []
for b in budgets:
    common = set.intersection(*[set(vals[(s, b)]) for s in systems_present]) if systems_present else set()
    for s in systems_present:
        sub = [q for q in common if label[q] == "synthesis"]
        if not sub:
            continue
        v = [vals[(s, b)][q] for q in sub]
        rng = random.Random(0)
        boots = sorted(sum(rng.choices(v, k=len(v))) / len(v) for _ in range(5000))
        rows.append({"scope": "synthesis", "system": s, "budget": b, "n": len(sub),
                     "mean": sum(v) / len(v), "ci_low": boots[624], "ci_high": boots[4374]})
    for base in BASELINES:
        if (base, b) not in vals:
            continue
        sub = [q for q in set(vals[(system, b)]) & set(vals[(base, b)]) if label[q] == "synthesis"]
        if not sub:
            continue
        diff = [vals[(system, b)][q] - vals[(base, b)][q] for q in sorted(sub)]
        rng = random.Random(0)
        boots = sorted(sum(rng.choices(diff, k=len(diff))) / len(diff) for _ in range(5000))
        rows.append({"scope": f"paired {system}-{base}", "system": system, "budget": b, "n": len(sub),
                     "mean": sum(diff) / len(diff), "ci_low": boots[624], "ci_high": boots[4374],
                     "better": sum(d > 0 for d in diff), "worse": sum(d < 0 for d in diff)})

out = REPO_ROOT / "results" / "tables" / f"{run.name}_synthesis_probe.csv"
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["scope", "system", "budget", "n", "mean", "ci_low", "ci_high", "better", "worse"])
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in w.fieldnames})
print(f"wrote {out.relative_to(REPO_ROOT)}")

print(f"\n{'scope':<26}{'B':>6}{'n':>5}{'answer':>9}{'95% CI':>18}")
for r in rows:
    ci = f"[{r['ci_low']:.3f}, {r['ci_high']:.3f}]"
    extra = f"  {r['better']}+/{r['worse']}-" if "better" in r else ""
    print(f"{r['scope']:<26}{r['budget']:>6}{r['n']:>5}{r['mean']:>9.3f}{ci:>18}{extra}")
if n_synth < 15:
    print(f"\nCAUTION: only {n_synth} synthesis questions in this cohort - any difference here is anecdote, not evidence.")
