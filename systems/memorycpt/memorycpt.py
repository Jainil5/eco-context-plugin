"""MemoryCPT-lite (reimpl.) of Zhang et al. 2026 (arXiv 2608.04843). No code or checkpoints were released.

Kept from the paper: offline query-agnostic construction of episodic + semantic memories; online RRF
(dense + BM25 k1=1.5, b=0.75, k_rrf=60) applied separately to the two stores, top-20 episodic and top-50
semantic; a query-aware summarizer whose output (<=512 tokens) is what the reader sees.
Not reproduced: the distilled LoRA-A construction model (replaced by prompting the configured LLM, with the
four roles collapsed into one call per session batch and no cross-batch EpisodeMerger) and the GRPO-trained
LoRA-B summarizer (replaced by a prompted, untrained summarizer). Details: docs/reimplementation_notes.md.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from core.data.longmemeval import Question
from core.llm.structured import parse_json_object, resolve_labels, session_batches

BUILD_PROMPT = """You are building a long-term memory store from a user's past chat sessions with an assistant.
Read the sessions below and output JSON with two lists:
- "episodes": one entry per coherent event or topic discussed. Each has "title", "summary" (2-4 sentences, \
concrete details, absolute dates), and "turns" (the [T#] labels it is based on).
- "semantic": durable facts about the user (preferences, possessions, relationships, plans, numbers, \
decisions) and important facts stated by the assistant. Each has "fact" (one self-contained sentence, \
absolute dates) and "turns" (the [T#] labels it is based on).
Resolve relative dates ("yesterday", "last week") against the session date. Output only the JSON object.

{sessions}"""


@dataclass
class Memory:
    kind: str                 # "episodic" | "semantic"
    text: str
    date: str                 # session date of the first cited turn (YYYY-MM-DD)
    source_turn_ids: list[str] = field(default_factory=list)


def construct(q: Question, llm, batch_tokens: int = 6000, parallel: int = 8) -> tuple[list[Memory], dict]:
    date_of = {t.turn_id: (t.session_date.strftime("%Y-%m-%d") if t.session_date else "unknown") for t in q.haystack}
    mems, stats = [], {"batches": 0, "unparsed_batches": 0, "dropped_no_provenance": 0}
    batches = session_batches(q.haystack, batch_tokens)
    def call(b):
        try:
            return llm.complete(BUILD_PROMPT.format(sessions=b.text), stage="construction", qid=q.qid,
                                max_tokens=16384)
        except Exception:  # noqa: BLE001 - a batch that fails after retries is counted, not fatal
            return None

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        replies = list(ex.map(call, batches))  # batches are independent; map keeps order
    for b, reply in zip(batches, replies):
        stats["batches"] += 1
        if reply is None:
            stats["failed_batches"] = stats.get("failed_batches", 0) + 1
            continue
        if reply.output_tokens >= 16384:  # hit the output cap: JSON is likely cut off
            stats["truncated_batches"] = stats.get("truncated_batches", 0) + 1
        data = parse_json_object(reply.text)
        if not isinstance(data, dict):
            stats["unparsed_batches"] += 1
            continue
        for kind, key, text_of in (("episodic", "episodes", lambda e: f"{e.get('title', '')}: {e.get('summary', '')}"),
                                   ("semantic", "semantic", lambda e: str(e.get("fact", "")))):
            for e in data.get(key) or []:
                if not isinstance(e, dict):
                    continue
                ids = resolve_labels(e.get("turns", []), b.labels)
                text = text_of(e).strip(": ")
                if not ids or not text:
                    stats["dropped_no_provenance"] += 1
                    continue
                mems.append(Memory(kind, text, date_of[ids[0]], ids))
    return mems, stats


SUMMARY_PROMPT = """Below are memories retrieved from a user's past conversations with an assistant, and a question \
the user is asking now. Write a concise summary containing only the information from these memories that is \
needed to answer the question: keep exact names, numbers and dates, and note when a fact was later updated. \
Do not answer the question. Keep the summary under {limit} tokens.

Current date: {date}
Question: {question}

Memories:
{memories}

Summary:"""


def _rrf_top(bm25, dense, query: str, qid: str, top: int) -> list[int]:
    from core.retrieval.rrf import rrf
    b = [i for i, _ in bm25.rank(query)][:top]
    d = [i for i, _ in dense.rank(query, qid)][:top]
    return [i for i, _ in rrf([b, d])][:top]


class MemoryCPTLite:
    name = "MemoryCPT-lite (reimpl.)"

    def __init__(self, services, top_episodic: int = 20, top_semantic: int = 50, summary_tokens: int = 512,
                 construction_parallel: int = 4):
        self.services, self.parallel = services, construction_parallel
        self.top_ep, self.top_sem, self.summary_tokens = top_episodic, top_semantic, summary_tokens

    def build_memory(self, q: Question):
        from core.retrieval.bm25 import BM25Index
        from core.retrieval.dense import DenseIndex
        mems, stats = construct(q, self.services.llm("extraction"), parallel=self.parallel)
        stores = {}
        for kind in ("episodic", "semantic"):
            ms = [m for m in mems if m.kind == kind]
            texts = [m.text for m in ms] or [""]
            stores[kind] = {"mems": ms, "bm25": BM25Index(texts, k1=1.5, b=0.75),
                            "dense": DenseIndex(texts, self.services.embedder, qid=q.qid)}
        return {"qid": q.qid, "stores": stores, "stats": stats}

    def build_context(self, query, memory, token_budget, query_date=None):
        from core.llm.tokenizer import get_tokenizer
        from core.context import ContextResult, header
        qid, tok = memory["qid"], get_tokenizer()
        picked = []
        for kind, top in (("episodic", self.top_ep), ("semantic", self.top_sem)):
            s = memory["stores"][kind]
            if s["mems"]:
                picked += [s["mems"][i] for i in _rrf_top(s["bm25"], s["dense"], query, qid, top)]
        head = header(query_date)
        limit = max(0, min(self.summary_tokens, token_budget - tok.count(head) - 1))
        summary = ""
        if picked and limit > 0:
            listing = "\n".join(f"- [{m.date}] ({m.kind}) {m.text}" for m in picked)
            date = query_date.strftime("%Y-%m-%d") if query_date else "unknown"
            summary = self.services.llm("query").complete(
                SUMMARY_PROMPT.format(limit=limit, date=date, question=query, memories=listing),
                stage="query_time", qid=qid, max_tokens=4096).text.strip()
        body = tok.truncate(summary, limit)
        context = head + body
        if tok.count(context) > token_budget:  # tokenizer merge at the join; shave the summary
            context = head + tok.truncate(summary, limit - (tok.count(context) - token_budget))
        n = tok.count(context)
        assert n <= token_budget, f"context {n} > budget {token_budget}"
        # Provenance = turns behind every memory handed to the summarizer (the summary itself is abstractive).
        ids = list(dict.fromkeys(i for m in picked for i in m.source_turn_ids))
        return ContextResult(context, n, ids, {
            "n_episodic": sum(m.kind == "episodic" for m in picked),
            "n_semantic": sum(m.kind == "semantic" for m in picked),
            "summary_tokens_raw": tok.count(summary), "summary_truncated": tok.count(summary) > limit,
            "construction": memory["stats"]})
