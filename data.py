import json
import random
from collections import Counter, defaultdict
from pathlib import Path

DATASET_DIR = Path("dataset")
SPLIT_FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def load_split(split: str) -> list[dict]:
    path = DATASET_DIR / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run python download_dataset.py --splits {split}")
    with open(path) as f:
        return json.load(f)


def save_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_sample(
    split: str = "s",
    n: int = 75,
    seed: int = 42,
    out_name: str = "longmemeval_sample.json",
) -> Path:
    data = load_split(split)
    rng = random.Random(seed)
    if n >= len(data):
        sample = data
    else:
        by_type = defaultdict(list)
        for ex in data:
            by_type[ex["question_type"]].append(ex)
        sample = []
        for qtype, items in sorted(by_type.items()):
            k = max(1, round(n * len(items) / len(data)))
            sample.extend(rng.sample(items, min(k, len(items))))
        if len(sample) > n:
            sample = rng.sample(sample, n)
        elif len(sample) < n:
            rest = [x for x in data if x not in sample]
            sample.extend(rng.sample(rest, min(n - len(sample), len(rest))))
    out = DATASET_DIR / out_name
    save_json(out, sample)
    return out


def export_conversation_samples(
    examples: list[dict],
    out_dir: Path | str = "samples",
    *,
    prefix: str = "conversation",
) -> Path:
    """Write one JSON file per example plus a manifest for comparison runs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, ex in enumerate(examples):
        short_id = ex["question_id"][:8]
        name = f"{prefix}_{i:03d}_{short_id}.json"
        path = out / name
        save_json(path, ex)
        manifest.append(
            {
                "file": name,
                "question_id": ex["question_id"],
                "question_type": ex["question_type"],
                "question": ex["question"],
                "answer": ex["answer"],
            }
        )
    manifest_path = out / "manifest.json"
    save_json(manifest_path, manifest)
    return manifest_path


def load_samples_dir(samples_dir: Path | str = "samples") -> list[dict]:
    root = Path(samples_dir)
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        examples = []
        for row in manifest:
            with open(root / row["file"]) as f:
                examples.append(json.load(f))
        return examples
    paths = sorted(root.glob("conversation_*.json"))
    if not paths:
        raise FileNotFoundError(f"no samples in {root}; run export_samples.py")
    out = []
    for path in paths:
        with open(path) as f:
            out.append(json.load(f))
    return out


def dataset_stats(data: list[dict]) -> dict:
    types = Counter(ex["question_type"] for ex in data)
    session_counts = [len(ex["haystack_sessions"]) for ex in data]
    turn_counts = [
        sum(len(s) for s in ex["haystack_sessions"]) for ex in data
    ]
    return {
        "n_examples": len(data),
        "question_types": dict(types),
        "sessions_per_example": {
            "min": min(session_counts),
            "max": max(session_counts),
            "mean": sum(session_counts) / len(session_counts),
        },
        "turns_per_example": {
            "min": min(turn_counts),
            "max": max(turn_counts),
            "mean": sum(turn_counts) / len(turn_counts),
        },
        "fields": list(data[0].keys()) if data else [],
    }
