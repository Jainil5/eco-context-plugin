"""YAML config loading: `extends:` chains + dotted CLI overrides, resolved to a plain dict."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = REPO_ROOT / "configs"


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _resolve(path: Path, seen: tuple[Path, ...] = ()) -> dict:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"Config extends cycle: {' -> '.join(map(str, seen + (path,)))}")
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent is None:
        return cfg
    return deep_merge(_resolve(path.parent / parent, seen + (path,)), cfg)


def parse_override(s: str) -> tuple[list[str], Any]:
    """'budgets=[500,1000]' / 'llm.reader.temperature=0' -> (keys, yaml-parsed value)."""
    key, _, raw = s.partition("=")
    if not _:
        raise ValueError(f"Override must be key=value, got {s!r}")
    return key.strip().split("."), yaml.safe_load(raw)


def load_config(path: str | Path = CONFIG_DIR / "base.yaml", overrides: list[str] | None = None) -> dict:
    cfg = _resolve(Path(path))
    for ov in overrides or []:
        keys, val = parse_override(ov)
        node = cfg
        for k in keys[:-1]:
            node = node.setdefault(k, {})
        node[keys[-1]] = val
    return cfg
