"""Single tokenizer used to enforce budgets for every method (rule 3)."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

# Keep the BPE files inside the repo cache so runs work offline after the first download.
os.environ.setdefault("TIKTOKEN_CACHE_DIR", str(Path(__file__).resolve().parents[2] / ".cache" / "tiktoken"))

import tiktoken  # noqa: E402

DEFAULT_ENCODING = "o200k_base"


@lru_cache(maxsize=8)
def _encoding(name: str) -> tiktoken.Encoding:
    return tiktoken.get_encoding(name)


class Tokenizer:
    def __init__(self, encoding: str = DEFAULT_ENCODING):
        self.name = encoding
        self._enc = _encoding(encoding)

    def encode(self, text: str) -> list[int]:
        return self._enc.encode(text, disallowed_special=())

    def count(self, text: str) -> int:
        return len(self.encode(text))

    def truncate(self, text: str, budget: int, keep: str = "head") -> str:
        """Cut `text` to at most `budget` tokens. keep="head" keeps the start, "tail" keeps the end.

        Decoding a token slice can re-tokenize to a different length at the cut, so we shrink
        until the re-encoded result fits.
        """
        if budget <= 0:
            return ""
        ids = self.encode(text)
        if len(ids) <= budget:
            return text
        n = budget
        while n > 0:
            piece = ids[:n] if keep == "head" else ids[-n:]
            out = self._enc.decode(piece)
            if self.count(out) <= budget:
                return out
            n -= 1
        return ""


_default: Tokenizer | None = None


def get_tokenizer(encoding: str = DEFAULT_ENCODING) -> Tokenizer:
    global _default
    if _default is None or _default.name != encoding:
        _default = Tokenizer(encoding)
    return _default


def count_tokens(text: str, encoding: str = DEFAULT_ENCODING) -> int:
    return get_tokenizer(encoding).count(text)
