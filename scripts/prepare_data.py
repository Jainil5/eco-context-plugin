"""M1: load LongMemEval-S, freeze the dev/test split (once), print statistics.
Usage: python scripts/prepare_data.py [key=value overrides]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config import REPO_ROOT, load_config  # noqa: E402
from core.data.longmemeval import load_longmemeval  # noqa: E402
from core.data.split import load_split, make_split  # noqa: E402
from core.data.stats import dataset_stats  # noqa: E402

cfg = load_config(overrides=sys.argv[1:])
data_path = REPO_ROOT / cfg["data"]["longmemeval_s"]
split_path = REPO_ROOT / cfg["data"]["split_file"]
qs = load_longmemeval(data_path)

if split_path.exists():
    split = load_split(split_path, data_path)
    print(f"Using frozen split {split_path.relative_to(REPO_ROOT)}")
else:
    split = make_split(qs, cfg["data"]["longmemeval_s"], cfg["data"]["dev_size"], cfg["seed"], split_path)
    print(f"Froze new split -> {split_path.relative_to(REPO_ROOT)}")

by_id = {q.qid: q for q in qs}
print(dataset_stats(qs, "LongMemEval-S all"))
print(dataset_stats([by_id[i] for i in split["dev"]], "dev"))
print(dataset_stats([by_id[i] for i in split["test"]], "test"))
