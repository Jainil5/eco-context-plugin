"""Creates results/runs/<run_id>/ with the resolved config and git commit (every run must)."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import yaml

from core.config import REPO_ROOT


def git_state() -> dict:
    def run(*args):
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
    return {"commit": run("rev-parse", "HEAD") or None, "dirty": bool(run("status", "--porcelain"))}


def new_run_dir(cfg: dict, name: str, root: str | Path | None = None) -> Path:
    root = Path(root or REPO_ROOT / cfg["paths"]["results"])
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    d = root / run_id
    (d / "traces").mkdir(parents=True, exist_ok=False)
    (d / "config.yaml").write_text(yaml.safe_dump({**cfg, "run_id": run_id, "git": git_state()}, sort_keys=False))
    return d
