import argparse
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
FILES = {
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
    "oracle": "longmemeval_oracle.json",
}


def download(name: str, out_dir: Path) -> Path:
    if name not in FILES:
        raise ValueError(f"unknown split {name}; choose from {list(FILES)}")
    url = f"{BASE}/{FILES[name]}"
    dest = out_dir / FILES[name]
    out_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000:
        print(f"skip {dest} (exists)")
        return dest
    print(f"downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"done {dest} ({dest.stat().st_size // 1_000_000} MB)")
    return dest


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--splits", nargs="+", default=["s", "m", "oracle"])
    p.add_argument("--out", type=Path, default=Path("dataset"))
    args = p.parse_args()
    for s in args.splits:
        download(s, args.out)
