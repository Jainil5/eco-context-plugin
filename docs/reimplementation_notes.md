# Reimplementation notes

Both baselines that need a model are reimplementations: no official code was released for either, and the papers
describe trained components we cannot train here. This file records what is kept, what is replaced, and what was
changed for reasons of correctness. It is written as a list of *deviations from the paper*, each with the reason, so
that a reader can tell which numbers are the paper's method and which are ours.

## ReFind (Li et al. 2026, arXiv 2608.12888) — `systems/refind/refind.py`

**Kept as described.** Two stages: a ReAct agent collects evidence, then the shared reader answers from what was
collected. Actions `search_chatrecord` / `take_note` / `finish_search`, at most 4 searches, BM25 (k1 = 1.2, b = 0.75,
Porter stemming, stopword removal) fused with session-level scores by RRF (k = 60), ±2-unit context expansion
around each hit, previously-returned sessions excluded from later searches, notes cut to the token budget in save
order (the method's own ranking is "the order it saved them in").

**Replaced.** The paper's agent model is a prompted GPT-class model; here the agent is the same model as the reader
(`compare.pool_models`, currently gpt-oss-20b). That is one shared model per run, which is what makes the comparison
fair, and it is the single biggest reason ReFind's numbers here are not the paper's numbers: the whole first stage is
instruction-following, and a 20B model follows the "search → take_note → search" rule imperfectly. `trace.invalid_actions`
and `trace.n_notes` are logged per question, so the note-taking rate is checkable rather than assumed.

**Deviations made for correctness, not for performance.** Each was found by reading a failing trace, and each only
gives ReFind *more* of a chance to score:

1. **Non-negative BM25 idf** (`core/retrieval/bm25.py`). `rank_bm25`'s Okapi idf is
   `log(N - df + 0.5) - log(df + 0.5)`, which is exactly 0 for a term occurring in half the documents and negative
   (floored at `epsilon x average_idf`) above that. The indexes in this repo are small — one document per session, or
   one per user/assistant unit, so 4-20 documents — and a question keyword that appears in half the sessions therefore
   scored nothing. Measured on the dev split: 110 of 777 question content tokens (14.2%) had idf exactly 0 in the
   oracle view, affecting 42.6% of questions; ReFind's agent got "No results." for the words it searched with. The
   Lucene form `log(1 + (N - df + 0.5) / (df + 0.5))` is used instead: same ordering in df, still punishes common
   terms, never blind. The same index is used by our own `bm25` / `hybrid` / `mmr` / ECO relevance path, so the fix is
   symmetric (it moved `eco_spans` *down* 0.027 at B = 250).
2. **Observation blocks do not repeat a unit.** The ±2 windows of two neighbouring hits overlapped and the same
   conversation was pasted into one observation up to 4 times: a probe on `2788b940` produced 96,698 characters of
   observation for a 51,970-character view. Blocks are now emitted disjoint within a search.
3. **`top_k` is capped at the paper's value (5).** The model may ask for any `top_k` in its Action Input; asking for
   10 meant one search returned the whole view, leaving the stage-2 budget cut nothing to do. Values are clamped to
   1..5, and the indices shown always match the blocks actually returned.
4. **`take_note` indices count across searches.** As first implemented, `take_note` could only address the *last*
   search's results. That is the paper's rule ("previous results become inaccessible after a new search"), but it
   interacts fatally with the seen-session exclusion: the follow-up search returns "No results." (every session was
   already shown), and the evidence the agent had just read became unsavable. One probe: search 1 returned the answer
   session, searches 2-4 returned nothing, the agent noted nothing, context = 25 characters, score 0. Numbering is
   now cumulative over the blocks shown so far, and the prompt rule says so. This is the change that turns ReFind from
   a 0.000 row into a baseline; it can only help ReFind, never hurt it.
5. **A blank agent reply ends collection instead of failing the question.** After the client's own retries, the model
   sometimes still answers with nothing (`finish_reason = stop`, empty text). Previously that raised and the whole
   job was lost, so ReFind's row came out with fewer questions than the others. Now the loop stops there and the
   reader answers from the notes saved so far, and `trace.blank_replies` counts it per question so the reader of the
   table can see how often the agent ran out of steam. Again this only gives ReFind more attempts, never fewer.

**Not changed, and worth knowing.** The oracle view hands ReFind a pool that *is* the evidence (4-21 units), so its
search step has no haystack to search; the whole haystack view (`compare.view=full`) is the setting its design
assumes. Numbers from the two views are not comparable and are reported separately.

## MemoryCPT-lite (Zhang et al. 2026, arXiv 2608.04843) — `systems/memorycpt/memorycpt.py`

**Kept as described.** Offline, query-agnostic construction of episodic + semantic memories; online RRF over dense
and BM25 (k1 = 1.5, b = 0.75, k_rrf = 60) applied separately to the two stores (top-20 episodic, top-50 semantic);
a query-aware summarizer whose output (≤ 512 tokens) is what the reader sees.

**Replaced.** The distilled LoRA-A construction model is replaced by prompting the configured LLM, with the paper's
four roles collapsed into one call per session batch and no cross-batch `EpisodeMerger`. The GRPO-trained LoRA-B
summarizer is replaced by a prompted, untrained summarizer. Both are stated in the module docstring; neither is a
reproduction of the paper's trained component, so the numbers here are "MemoryCPT-shaped pipeline with an untrained
model", which is the honest label for the row.

## Shared harness deviations that apply to every row

- **Judge budget.** LongMemEval's official judge prompt is answered with `max_tokens = 10`; gpt-oss reasons before
  answering and returns an empty reply at 10, so the judge gets 1024 (`configs/compare.yaml`
  `judge_max_tokens`). Verdict parsing is unchanged (`'yes'` in the reply).
- **`reasoning_effort: low` on every chat call** (`llm_call_params`). With default effort, the NVIDIA NIM route
  answers `openai/gpt-oss-20b` requests with HTTP 200 and an *empty* completion carrying
  `response_metadata.error` (`unexpected tokens remaining in message header`). The wrapper used to accept that as a
  successful empty reply and cache it, which is how a whole system's row came out 0.000 with 25 characters of context.
  `core/llm/client.py` now raises on a provider error inside the message metadata (fatal, not retried) and on an
  empty completion (retried as transient), treats a cached blank as a miss so poisoned entries heal, and folds
  per-run default params into the cache key.
- **Long prompts still come back blank, even at low effort.** A ReFind observation is a ~50K-character dump, and on
  those calls NIM returns an empty completion with `finish_reason = stop` — no metadata error to raise on. The client
  retries four times and then gives up, and `integrate.api.nvidia.com` also times out reads at 300 s on some of these
  calls (observed: 2 of 12 ReFind questions lost to `EmptyResponseError`, 1 to `ReadTimeout`, in a 12-question run
  under 4-way concurrency). Both are provider limits, not method behaviour, so a baseline's row can be short on
  questions the cheap retrieval rows never miss; `results/runs/<run_id>/errors.jsonl` records exactly which ones, and
  reruns hit the cache so only the missing jobs cost anything.
