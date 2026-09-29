import csv
import json

from core.eval.compare import COLUMNS, build_rows, load_runs, write_csv


def _run(tmp_path, name, preds, costs=()):
    d = tmp_path / name
    d.mkdir()
    (d / "predictions.jsonl").write_text("".join(json.dumps(p) + "\n" for p in preds))
    (d / "costs.jsonl").write_text("".join(json.dumps(c) + "\n" for c in costs))


def test_ranks_best_method_per_budget_and_excludes_judge_cost(tmp_path):
    P = lambda m, b, q, t, ok: {"method": m, "budget": b, "qid": q, "qtype": t, "correct": ok,
                                "n_tokens": b, "evidence_recall": 1.0 if ok else 0.0}
    _run(tmp_path, "r1", [P("eco", 500, "q1", "temporal", True), P("eco", 500, "q2", "multi", True),
                         P("refind", 500, "q1", "temporal", False), P("refind", 500, "q2", "multi", True)],
         [{"method": "eco", "qid": "q1", "stage": "construction", "input_tokens": 100, "output_tokens": 0, "usd": 0.01},
          {"method": "eco", "qid": "q1", "stage": "judge", "input_tokens": 999, "output_tokens": 0, "usd": 9.0}])
    rows = build_rows(*load_runs(tmp_path))
    top = {(r["method"], r["qtype"]): r for r in rows if r["budget"] == 500}
    assert top[("eco", "all")]["accuracy"] == 1.0 and top[("refind", "all")]["accuracy"] == 0.5
    assert top[("refind", "all")]["best_at_budget"] == "eco" and top[("refind", "all")]["delta_vs_best"] == -0.5
    assert top[("eco", "all")]["construction_tokens_per_q"] == 50  # 100 over 2 questions, judge excluded
    assert top[("eco", "all")]["system_usd_per_q"] == 0.005
    out = write_csv(rows, tmp_path / "t.csv")
    assert next(csv.reader(out.open())) == COLUMNS


def test_no_runs_gives_header_only(tmp_path):
    out = write_csv(build_rows(*load_runs(tmp_path)), tmp_path / "t.csv")
    assert out.read_text().strip() == ",".join(COLUMNS)
