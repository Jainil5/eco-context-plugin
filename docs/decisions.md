# Decision log

Newest last, because this is a record of what was decided and when, not a summary. Each entry names the evidence that
forced it. Numbers quoted here resolve to files under `results/tables/` or to the commits cited.

## D-001 · 2026-10-22 · Empty provider replies are failures, not results

`deef0e8`. The NIM routes return HTTP 200 with an error folded into `response_metadata`, and blank completions with
`finish_reason='stop'` on long prompts. The client had been logging those as successful answers, which is how
ReFind got published at 0/12 with 11 tokens of context — the 11 tokens were only the date header. Now:
`ProviderError` is fatal and never retried, `EmptyResponseError` is transient and retried, a blank completion after
retries aborts the loop instead of poisoning the table, and a cached blank is treated as a cache miss so old poisoned
entries heal themselves.

Consequence for reading the repo: **any baseline number older than this commit is untrustworthy**, including the
superseded 2026-09-30 pilot kept in `README.md` as the honest version of the record.

## D-002 · 2026-10-22 · BM25 idf fixed, and it moved ECO down

`be9c259`. `rank_bm25`'s `BM25Okapi` gives idf 0 (and negative below a threshold) for terms appearing in more than half
the corpus: 110 of 777 dev question tokens (14.2%) scored 0.0, touching 42.6% of questions. Replaced with Lucene's
`log(1 + (N − df + 0.5)/(df + 0.5))`. Every published table was refreshed to post-fix state (Tier-1
`20261022-112348`, ablations `20261022-123158`, span sweep `20261022-120213`).

Consequence: the fix made the retrieval baselines *stronger* and ECO's relative standing *worse* (−0.027
`answer_in_ctx` at B=250). Refreshing rather than keeping flattering numbers was the decision, not the arithmetic.

## D-003 · 2026-10-22 · The presentation unit is the whole system

`bb59c26`. Variant alphabets (`eco_sup`, `eco_breadth`, `eco_facets`, …) became appendix material; the headline is one
row per system against one shared reader and one shared judge. The n=12 pilot (`20261022-133443`) is reported with its
cost columns and with the top row belonging to **MemoryCPT-lite (9/12), not ECO (8/12)**.

## D-004 · 2026-10-24 → 2026-10-25 · The architecture claim was pre-registered and it failed

Written into `configs/compare.yaml` before the runs: the zero-LLM facility-location pipeline is Pareto-equal to the LLM
systems at equal budget. PRIMARY: `eco_spans` indistinguishable from BOTH `eco_facts` and `memorycpt` at B=250 and
B=500; SECONDARY: beats `refind` with the CI excluding zero; FAIL: either LLM system ahead with a CI excluding zero at
both budgets.

Run `20261025-164642` (7 systems × 3 budgets × 94 questions = 1,974 rows, zero errors):

- **PRIMARY failed.** vs `memorycpt` at B=250: **−0.181 [−0.287, −0.075]**. At B=500 −0.106 [−0.223, +0.011], at
  B=1000 −0.053 [−0.149, +0.043]. vs `eco_facts` every CI covers zero. The literal FAIL clause did not trigger
  (`memorycpt` is significant at only one budget), but the claim as written needs indistinguishability at both.
- **SECONDARY held**: +0.383 / +0.372 / +0.426 vs `refind`, all CIs excluding zero — against a baseline measured with
  the deviation-6 auto-save fallback enabled (it fired on 108/282 contexts, 38%).
- Among the zero-LLM systems it is a **plateau**: `eco_spans`, `bm25`, `span_hybrid` all 0.617 at B=250, paired diff
  exactly 0.

Decision: the zero-LLM selection claim is withdrawn as a headline. It is not replaced with a softer sentence about the
same benchmark.

## D-005 · 2026-10-25 · The gap is compression, not weighting

`memorycpt` reaches 0.798/0.809/0.819 on **96–127 reader tokens** with 4.7 LLM calls per question; `eco_spans` reaches
0.617/0.702/0.766 on 236–964 with none. Even `eco_facts`, with 1.7 offline build calls, only gets to 0.734 at B=500.
So at tight budgets the winner is the system that *condenses*, and no re-weighting of a selection over verbatim spans
closes that. Two corroborating nulls: the free synthesis attempt (`eco_derive`, templated count/date aggregates) is a
**loss** at B=250 (−0.053 [−0.106, −0.011] vs `eco_spans`) and its losses concentrate on knowledge-update questions,
because an aggregate flattens the old→new distinction; and `scripts/synthesis_probe.py` shows 39 of the cohort are
**synthesis-required** — the answer text appears in no turn at all, which no selector can retrieve.

Decision: Phase 3's learned `sim(i,m)` is not built as a LongMemEval re-weighting. The learned term is re-scoped to
staleness/supersession inside the coding-agent benchmark.

## D-006 · 2026-10-25 · Mechanisms are kept behind flags even when they are null

`73453bc`, `9db6574`. Supersession γ, hard and soft breadth, facets, adaptive tau, derived facts — all default off, all
with their counters kept in `trace_metrics.jsonl` so a null can be checked to be a real null rather than a mechanism
that never fired. In the oracle view the mechanism's precondition barely exists (`sessions_in_pool` 1.91,
`breadth_queries` 0.63), and the n=8 full-view probe (`20261024-010714`) is also null with the conditions present
(22.6 sessions in pool, 50 superseded span pairs per question, every variant's `answer_in_ctx` equal to frozen).

## D-007 · 2026-11-02 · New implementation goes on `research/budgeted-coverage-v2`; results stay where they are

Owner's instruction: existing results must be saved and preserved, and the plan is a new implementation. Actions:

- `main` tip `3df5ca5` tagged **`archive/n94-architecture-verdict`**. 124 tables + 4 figures are tracked in git at that
  commit, so the published record is durable independent of any sandbox (the working sandbox had already been reclaimed
  once, losing `.cache`, `dataset/` and `results/runs/`).
- New branch `research/budgeted-coverage-v2` created from the same commit. Experiments for the new plan land here and
  merge back only with their tables and a decided claim.
- `docs/` is un-ignored in `.gitignore` (it previously shipped only `reimplementation_notes.md`), which is what makes
  `implementation_plan.md`, this log, and later `reproduction.md`/`benchmark_card.md` committable at all.

## D-008 · 2026-11-02 · Phase 1's exit gate is already answered, so Phase 1 does not run as written

The plan's gate was "if ECO beats hybrid at B ≥ 500 with a CI excluding zero, keep the accuracy headline; else reframe
to recall-and-cost and move the learned component to the foreground". On dev n=94 the comparison is `eco_spans` 0.702
vs `span_hybrid` 0.713 and `bm25` 0.702 at B=500 — no CI excluding zero, one of them behind. The "else" branch applies
and D-005 says the reframe cannot be about LongMemEval selection either. Two things must be settled before paid runs:
a reader stronger than `gpt-oss-20b` with a **different** judge, and a power analysis that shrinks the ≈19,200-row grid
to an affordable pre-registered subset.
