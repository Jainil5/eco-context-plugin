# Implementation Plan v2 — Budgeted Coverage Context Selection

Status: **planning record for `research/budgeted-coverage-v2`.** Nothing on this branch has been run yet.

Provenance: this is the owner's plan as stated in session, re-written into the repo after the previous session's
context was compacted. It is a faithful reconstruction of the structure and the constraints, not a byte-for-byte copy
of the original text. If the original is pasted again it should replace this file verbatim, with this note moved to
`docs/decisions.md`.

## Why this branch exists

`main` (tip `3df5ca5`, preserved as tag `archive/n94-architecture-verdict`) carries the published record: 124 tables
and 4 figures, every number quoted in `README.md` resolvable to a file. Its last substantive entry is a **failed
pre-registered claim** — see `docs/decisions.md` D-004/D-005. This branch is where new implementation happens, so
that later experiments cannot move the published numbers. Work here merges back only when a claim is decided, with the
tables that decide it.

## Ground rules carried over (non-negotiable)

1. **Whole-system claims.** The headline comparison is system vs system — one row each for ECO, ReFind, MemoryCPT,
   retrieval — with a shared reader and a shared judge. Per-mechanism variant tables are appendix material, never the
   headline.
2. **Pre-register before running**, in `configs/compare.yaml` or here, with the failure condition written down first.
3. **Report misses as misses.** No re-framing a null into a tie, and no claiming a win the control contradicts.
4. **Cost is a column, not a footnote**: chat-model tokens for construction and per query, alongside context tokens
   actually handed to the reader.
5. **Paired statistics only**: bootstrap CI over the same questions plus exact McNemar (`scripts/mcnemar.py`), and a
   power analysis before any grid is committed to.

## Phase status against the repo

| Phase | What it asked for | State | Evidence / what is missing |
|---|---|---|---|
| 0 | Reproduce the existing pipeline end to end | **done** | `README.md` run section; Tier-1 `20261022-112348`, whole-system Tier-2 `20261022-133443`, n=94 `20261025-164642` |
| 1 | Full-test QA, stronger reader, judge ≠ reader, both views, B 250–2000, judge human-validation, exit gate | **gate already answered "else"** | dev n=94: `eco_spans` 0.702 vs `span_hybrid` 0.713 and `bm25` 0.702 at B=500, no CI excluding zero. Test-split run and the reader/judge change are **not** done (blocked on API key, see Open items) |
| 2a | Better segmentation unit | **done as sentence spans** | `systems/eco/spans.py`; +0.090 `answer_in_ctx` over turn-level ECO at B=250 |
| 2b | Hybrid warm start (`warm_start.m`) | **not built** | the only Phase-2 item still open on the selection side |
| 2c | Recency / supersession | **built, measured null** | `γ` in `systems/eco/objective.py`; `superseded_pairs` fires (87.4/question in oracle view, 50 in the n=8 full-view probe) with paired diff exactly 0.000 |
| 2d | Re-tune `tau` + `sim_floor` | **done, tie** | `scripts/sweep_spans.py` over 18 cells: no tau or floor is consistently best |
| 3 | Learned `sim(i,m)` (bi-encoder adapter on frozen `nomic-embed-text`, CV over question types, 3 seeds, leakage check) | **re-scoped, not built** | See D-005: the measured gap is compression, not re-weighting. Learn the *staleness/supersession* term inside the coding-agent setting instead, where the failure mode is real |
| 4 | Coding-agent benchmark (SWE-bench Lite, base-commit leakage control, 50 dev / 100 test, AST staleness, fixed harness; conditions no-memory / full-history / top-k-hybrid / ECO / ECO+learned+staleness) | **not started — this is now the primary contribution** | needs `systems/coding/`, `scripts/build_coding_bench.py`, `scripts/run_coding.py`, `tests/test_leakage.py` |
| 5 | LoCoMo, sensitivity, cost, failure analysis | not started | failure analysis tooling already exists (`scripts/analyze_failures.py` if added; per-question counters are in `trace_metrics.jsonl`) |
| 6 | Write-up / release | not started | `docs/reproduction.md`, `docs/benchmark_card.md` are unblocked: `docs/` is now tracked |

## What changed about the headline

The old headline was "budgeted coverage selection beats retrieval on LongMemEval-S". It does not survive the n=94
run. What the data supports instead:

- At **equal budget and equal cost** (zero LLM calls), selection is a plateau: `eco_spans`, `span_hybrid`, `bm25` are
  indistinguishable, and all three tie at 0.617 at B=250.
- Accuracy in this view is bought with **LLM calls at construction time** (`memorycpt` 0.798/0.809/0.819 on 96–127
  reader tokens, 4.7 calls/question; `eco_facts` 0.734 at B=500 with 1.7 build calls), not by a better objective.
- Zero-LLM selection is decisively better than the **agentic** baseline (`refind` 0.234/0.330/0.340, +0.38/+0.37/+0.43,
  every CI excluding zero) — but that measures an agent that gave up on 38% of contexts, so it is a weak win.

Therefore the next contribution has to **condense evidence without spending an LLM call per question**, or move to a
setting where the thing coverage selects for actually fails. Phase 4 does the latter: a coding agent's context has real
stale-winner failures, thousands of candidate units, and a budget that bites. Phase 3's learned similarity becomes the
staleness/supersession term in that objective rather than a standalone re-weighting of LongMemEval spans.

## Open items that block paid work

1. Reader and judge must both be stronger than `gpt-oss-20b` and must not be the same model. Every published number
   uses `gpt-oss-20b` for both roles, which is a known validity threat, not a neutral choice.
2. That needs a paid key. The NVIDIA key pasted into chat (`nvapi-…`) is disclosed in plaintext and **must be
   rotated** before anything else is run against it.
3. The Phase-1 grid as written is ≈ 400 questions × 6 systems × 4 budgets × 2 views ≈ 19,200 rows against the 1,974
   rows that produced the n=94 verdict. A pre-registered affordable subset and a power analysis come first; the full
   grid does not.
4. This sandbox was reclaimed: no `.cache`, no `dataset/`, no `.venv`, no running Ollama. Any rerun re-embeds from
   zero, so the frozen-cache experiments are cheap to reproduce only on the machine that has them.
