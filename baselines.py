import math
import re

from context_format import fit_chunks_to_budget, format_sessions, session_chunks
from model import embedding, llm
from tokens import count_tokens, truncate_to_budget

METHOD_NAMES = [
    "full_history",
    "recent_window",
    "rolling_summary",
    "refind",
    "memorycpt",
    "eco",
]

# Normal agent: full chat history (truncated to budget).
BASELINE_AGENT = "full_history"
# ECO V0: retrieve user turns by question relevance + recency.
ECO_AGENT = "eco"


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def full_history(entry: dict, budget: int, **_kwargs) -> str:
    text = format_sessions(entry["haystack_dates"], entry["haystack_sessions"])
    return truncate_to_budget(text, budget)


def recent_window(entry: dict, budget: int, **_kwargs) -> str:
    chunks = [c[1] for c in session_chunks(entry)]
    return fit_chunks_to_budget(chunks, budget)


def rolling_summary(entry: dict, budget: int, **_kwargs) -> str:
    chunks = session_chunks(entry)
    if not chunks:
        return ""
    summary = ""
    for _, block in chunks:
        if not summary:
            summary = block
            continue
        if count_tokens(summary) + count_tokens(block) <= budget:
            summary = f"{summary}\n\n{block}"
            continue
        prompt = (
            "Merge the previous summary with the new chat. "
            "Keep facts, dates, names, numbers. Be concise.\n\n"
            f"Previous summary:\n{summary}\n\nNew chat:\n{block}\n\nUpdated summary:"
        )
        summary = llm.invoke(prompt).content.strip()
    return truncate_to_budget(summary, budget)


def _retrieve_ranked(
    entry: dict,
    budget: int,
    *,
    user_only: bool = False,
) -> str:
    question = entry["question"]
    chunks = session_chunks(entry, user_only=user_only)
    if not chunks:
        return ""
    texts = [c[1] for c in chunks]
    q_emb = embedding.embed_query(question)
    doc_embs = embedding.embed_documents(texts)
    n = len(chunks)
    scored = []
    for i, ((idx, text), emb) in enumerate(zip(chunks, doc_embs)):
        recency = (i + 1) / n
        scored.append((0.75 * _cosine(q_emb, emb) + 0.25 * recency, idx, text))
    scored.sort(key=lambda x: x[0], reverse=True)
    ordered = []
    used = 0
    for _, idx, text in scored:
        t = count_tokens(text)
        if used + t > budget:
            if not ordered:
                ordered.append(truncate_to_budget(text, budget))
            break
        ordered.append((idx, text))
        used += t
    ordered.sort(key=lambda x: x[0])
    return "\n\n".join(t for _, t in ordered)


def refind(entry: dict, budget: int, **_kwargs) -> str:
    return _retrieve_ranked(entry, budget, user_only=False)


def eco(entry: dict, budget: int, **_kwargs) -> str:
    """ECO V0: relevance-ranked retrieval over user messages (less noise than full history)."""
    return _retrieve_ranked(entry, budget, user_only=True)


def memorycpt(entry: dict, budget: int, **_kwargs) -> str:
    question = entry["question"]
    chunks = session_chunks(entry)
    if not chunks:
        return ""
    lines = []
    for j, (idx, text) in enumerate(chunks):
        preview = text.replace("\n", " ")[:220]
        lines.append(f"{j}: {preview}")
    prompt = (
        "Pick session indices relevant to answering the question. "
        "Reply with comma-separated numbers only.\n\n"
        f"Question: {question}\n\nSessions:\n"
        + "\n".join(lines)
        + "\n\nIndices:"
    )
    raw = llm.invoke(prompt).content
    picked = [int(x) for x in re.findall(r"\d+", raw)]
    picked = [i for i in picked if 0 <= i < len(chunks)]
    if not picked:
        return refind(entry, budget)
    texts = []
    used = 0
    for j in sorted(set(picked)):
        text = chunks[j][1]
        t = count_tokens(text)
        if used + t > budget:
            if not texts:
                texts.append(truncate_to_budget(text, budget))
            break
        texts.append(text)
        used += t
    return "\n\n".join(texts)


BUILDERS = {
    "full_history": full_history,
    "recent_window": recent_window,
    "rolling_summary": rolling_summary,
    "refind": refind,
    "memorycpt": memorycpt,
    "eco": eco,
}
