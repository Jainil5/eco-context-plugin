import argparse
import json
from pathlib import Path

from data import export_conversation_samples, load_split


def load_source(split: str) -> list[dict]:
    if split == "sample":
        path = Path("dataset/longmemeval_sample.json")
        with open(path) as f:
            return json.load(f)
    return load_split(split)


def main():
    p = argparse.ArgumentParser(
        description="Copy conversation examples from the dataset into samples/"
    )
    p.add_argument("--split", default="sample", help="s | m | oracle | sample")
    p.add_argument("--n", type=int, default=8, help="number of conversations to export")
    p.add_argument("--offset", type=int, default=0, help="skip first N examples")
    p.add_argument("--out-dir", default="samples")
    args = p.parse_args()
    data = load_source(args.split)
    slice_ = data[args.offset : args.offset + args.n]
    if not slice_:
        raise SystemExit("no examples selected; check --offset and --n")
    manifest = export_conversation_samples(slice_, args.out_dir)
    print(f"wrote {len(slice_)} conversations to {args.out_dir}/")
    print(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
