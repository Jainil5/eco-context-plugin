import pytest
import yaml

from core.config import CONFIG_DIR, load_config
from core.rundir import new_run_dir
from core.llm.cost import (CostLimitExceeded, Pricing, PlannedCalls, UnknownPriceError,
                          cost_guard, estimate_usd)

P = Pricing({"m": {"input_per_1m": 1.0, "output_per_1m": 4.0}})


def test_estimate_usd():
    plan = [PlannedCalls("m", 1000, 1000, 100)]  # 1M in, 100K out
    assert estimate_usd(plan, P) == pytest.approx(1.0 + 0.4)


def test_cost_guard_blocks_over_budget():
    out = []
    with pytest.raises(CostLimitExceeded):
        cost_guard([PlannedCalls("m", 1000, 1000, 100)], P, max_usd=1.0, printer=out.append)
    assert any("TOTAL" in l for l in out), "estimate must be printed before refusing"


def test_cost_guard_passes_under_budget():
    assert cost_guard([PlannedCalls("m", 10, 100, 10)], P, max_usd=1.0, printer=lambda *_: None) < 1.0


def test_cost_guard_unknown_model_fails_safe():
    with pytest.raises(UnknownPriceError):
        cost_guard([PlannedCalls("mystery", 1, 1, 1)], P, max_usd=100, printer=lambda *_: None)


def test_cost_guard_no_limit_prints_and_never_blocks():
    out = []
    plan = [PlannedCalls("m", 10**6, 10**4, 10**3), PlannedCalls("mystery", 5, 5, 5)]
    total = cost_guard(plan, P, max_usd=None, printer=out.append)
    assert total > 1000 and any("unpriced" in l for l in out) and any("no max_usd" in l for l in out)


def test_base_config_loads():
    cfg = load_config()
    assert cfg["tokenizer"] == "o200k_base"
    assert cfg["budgets"] == [500, 1000, 2000, 4000, 8000]
    assert cfg["max_usd"] is None
    for role in ("reader", "judge", "extraction", "embeddings"):
        assert cfg["models"][role] in cfg["pricing"], f"{role} model needs a price"


def test_extends_and_overrides(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text(yaml.safe_dump({"a": 1, "nested": {"x": 1, "y": 2}, "budgets": [1, 2]}))
    child = tmp_path / "child.yaml"
    child.write_text(yaml.safe_dump({"extends": "base.yaml", "nested": {"y": 3}}))
    cfg = load_config(child, ["budgets=[500]", "nested.z=true", "a=2.5"])
    assert cfg == {"a": 2.5, "nested": {"x": 1, "y": 3, "z": True}, "budgets": [500]}


def test_extends_cycle_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError):
        load_config(tmp_path / "a.yaml")


def test_run_dir_records_config_and_git(tmp_path):
    d = new_run_dir(load_config(CONFIG_DIR / "base.yaml"), "smoke", root=tmp_path)
    saved = yaml.safe_load((d / "config.yaml").read_text())
    assert (d / "traces").is_dir()
    assert saved["run_id"].endswith("_smoke") and "commit" in saved["git"] and "max_usd" in saved
