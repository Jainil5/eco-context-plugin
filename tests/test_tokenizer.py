import random

import pytest

from core.llm.tokenizer import Tokenizer, count_tokens

TEXTS = [
    "",
    "hello world",
    "User: I moved to Berlin on 2023-05-01. 🚀 Unicode — ümlauts, 中文字符, and emojis 👩‍👩‍👧",
    " ".join(f"word{i}" for i in range(3000)),
]


@pytest.mark.parametrize("text", TEXTS)
@pytest.mark.parametrize("budget", [0, 1, 5, 17, 100, 500, 10_000])
@pytest.mark.parametrize("keep", ["head", "tail"])
def test_truncate_never_exceeds_budget(text, budget, keep):
    tok = Tokenizer()
    out = tok.truncate(text, budget, keep=keep)
    assert tok.count(out) <= budget
    if tok.count(text) <= budget:
        assert out == text


def test_truncate_random_unicode_fuzz():
    tok = Tokenizer()
    rng = random.Random(0)
    alphabet = "abc xyz 0123 éü中文👩‍👧\n.,"
    for _ in range(200):
        text = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        b = rng.randint(0, 120)
        assert tok.count(tok.truncate(text, b, keep=rng.choice(["head", "tail"]))) <= b


def test_head_and_tail_keep_correct_side():
    tok = Tokenizer()
    text = " ".join(f"w{i}" for i in range(200))
    assert text.startswith(tok.truncate(text, 20, "head"))
    assert text.endswith(tok.truncate(text, 20, "tail"))


def test_default_encoding_is_o200k():
    assert Tokenizer().name == "o200k_base"
    assert count_tokens("hello world") == 2
