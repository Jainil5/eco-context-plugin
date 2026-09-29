"""Frozen dev/test split, stratified by question type (+ abstention flag), fixed seed (rule 2)."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

from core.data.longmemeval import Question


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stratified_dev(questions: list[Question], dev_size: int, seed: int) -> list[str]:
    """Proportional allocation per stratum (largest remainder), random pick inside each stratum."""
    by: dict[str, list[str]] = defaultdict(list)
    for q in questions:
        by[q.stratum].append(q.qid)
    n = len(questions)
    exact = {s: dev_size * len(ids) / n for s, ids in by.items()}
    alloc = {s: int(x) for s, x in exact.items()}
    for s in sorted(exact, key=lambda s: (-(exact[s] - alloc[s]), s))[: dev_size - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    dev = []
    for s in sorted(by):
        dev += rng.sample(sorted(by[s]), alloc[s])
    return sorted(dev)


def make_split(questions: list[Question], dataset_path: str | Path, dev_size: int, seed: int,
               out: str | Path, force: bool = False) -> dict:
    out = Path(out)
    if out.exists() and not force:
        raise FileExistsError(f"{out} is frozen; refusing to overwrite (pass force=True only if you mean it).")
    dev = stratified_dev(questions, dev_size, seed)
    dev_set = set(dev)
    split = {
        "dataset": str(dataset_path), "dataset_sha256": file_sha256(dataset_path),
        "seed": seed, "stratify_by": "question_type + abstention",
        "dev": dev, "test": sorted(q.qid for q in questions if q.qid not in dev_set),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, indent=1))
    return split


def load_split(path: str | Path, dataset_path: str | Path | None = None) -> dict:
    split = json.loads(Path(path).read_text())
    if dataset_path and file_sha256(dataset_path) != split["dataset_sha256"]:
        raise ValueError(f"Dataset {dataset_path} does not match the file the split was frozen on.")
    return split
