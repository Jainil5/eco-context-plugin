from tokens import count_tokens, truncate_to_budget


def format_turn(turn: dict) -> str:
    role = turn["role"].capitalize()
    return f"{role}: {turn['content']}"


def format_session(date: str, session: list[dict], user_only: bool = False) -> str:
    turns = session
    if user_only:
        turns = [t for t in session if t["role"] == "user"]
    if not turns:
        return ""
    body = "\n".join(format_turn(t) for t in turns)
    return f"[{date}]\n{body}"


def format_sessions(
    dates: list[str],
    sessions: list[list[dict]],
    indices: list[int] | None = None,
    user_only: bool = False,
) -> str:
    if indices is None:
        indices = list(range(len(sessions)))
    blocks = []
    for i in indices:
        block = format_session(dates[i], sessions[i], user_only=user_only)
        if block:
            blocks.append(block)
    return "\n\n".join(blocks)


def session_chunks(entry: dict, user_only: bool = False) -> list[tuple[int, str]]:
    chunks = []
    for i, (date, session) in enumerate(
        zip(entry["haystack_dates"], entry["haystack_sessions"])
    ):
        text = format_session(date, session, user_only=user_only)
        if text:
            chunks.append((i, text))
    return chunks


def fit_chunks_to_budget(chunks: list[str], budget: int) -> str:
    if budget <= 0:
        return ""
    selected = []
    used = 0
    for block in reversed(chunks):
        t = count_tokens(block)
        if used + t > budget and selected:
            break
        if t > budget:
            block = truncate_to_budget(block, budget - used)
            t = count_tokens(block)
        if used + t > budget:
            break
        selected.append(block)
        used += t
    selected.reverse()
    return "\n\n".join(selected)
