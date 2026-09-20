import tiktoken

_enc = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_enc.encode(text))


def truncate_to_budget(text: str, budget: int) -> str:
    ids = _enc.encode(text)
    if len(ids) <= budget:
        return text
    return _enc.decode(ids[-budget:])
