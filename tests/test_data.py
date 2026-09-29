import json
from datetime import datetime

import pytest

from core.data.longmemeval import load_longmemeval, normalize, parse_date
from core.data.split import load_split, make_split, stratified_dev


def raw(qid, qtype="multi-session", n_sess=2):
    return {
        "question_id": qid, "question_type": qtype, "question": "q?", "question_date": "2023/05/30 (Tue) 23:40",
        "answer": 7, "answer_session_ids": ["s1"],
        "haystack_session_ids": [f"s{i}" for i in range(n_sess)],
        "haystack_dates": ["2023/05/20 (Sat) 02:21"] * n_sess,
        "haystack_sessions": [[{"role": "user", "content": f"hi {i}"},
                               {"role": "assistant", "content": "ok", **({"has_answer": True} if i == 1 else {})}]
                              for i in range(n_sess)],
    }


def test_parse_date():
    assert parse_date("2023/05/20 (Sat) 02:21") == datetime(2023, 5, 20, 2, 21)
    assert parse_date(None) is None


def test_normalize():
    q = normalize(raw("abc_abs"))
    assert q.answer == "7" and q.is_abstention and q.stratum == "multi-session_abs"
    assert [t.turn_id for t in q.haystack] == ["s0:0", "s0:1", "s1:0", "s1:1"]
    assert q.evidence_turn_ids == ["s1:1"]
    assert q.haystack[0].has_answer is None


def test_duplicate_sessions_get_unique_turn_ids():
    r = raw("x"); r["haystack_session_ids"] = ["s0", "s0"]
    q = normalize(r)
    assert [t.turn_id for t in q.haystack] == ["s0:0", "s0:1", "s0#2:0", "s0#2:1"]


def test_split_stratified_deterministic_and_frozen(tmp_path):
    data = [raw(f"a{i}", "A") for i in range(60)] + [raw(f"b{i}", "B") for i in range(30)] + \
           [raw(f"c{i}_abs", "B") for i in range(10)]
    p = tmp_path / "d.json"; p.write_text(json.dumps(data))
    qs = load_longmemeval(p)
    dev = stratified_dev(qs, 20, seed=1)
    assert dev == stratified_dev(qs, 20, seed=1) and len(dev) == 20
    assert sum(d.startswith("a") for d in dev) == 12 and sum(d.startswith("c") for d in dev) == 2
    out = tmp_path / "split.json"
    s = make_split(qs, p, 20, 1, out)
    assert set(s["dev"]).isdisjoint(s["test"]) and len(s["dev"]) + len(s["test"]) == 100
    with pytest.raises(FileExistsError):
        make_split(qs, p, 20, 2, out)
    assert load_split(out, p)["dev"] == dev
    p.write_text(json.dumps(data[:-1]))
    with pytest.raises(ValueError):
        load_split(out, p)


def test_oracle_view_keeps_only_evidence_sessions():
    from core.data.views import apply_view
    r = raw("x", n_sess=4)
    r["haystack_session_ids"][3] = "s1"  # repeated evidence session -> "s1#2", also evidence
    q = normalize(r)
    o = apply_view(q, "oracle")
    assert {t.session_id for t in o.haystack} == {"s1", "s1#2"}
    assert set(q.evidence_turn_ids) <= {t.turn_id for t in o.haystack}
    assert [t.turn_id for t in o.haystack] == [t.turn_id for t in q.haystack if t.session_id in ("s1", "s1#2")]
    assert apply_view(q, "full") is q
    with pytest.raises(ValueError):
        apply_view(q, "tiny")
