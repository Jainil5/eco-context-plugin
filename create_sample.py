import argparse

from data import create_sample, dataset_stats
import json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="s")
    p.add_argument("--n", type=int, default=75)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = create_sample(split=args.split, n=args.n, seed=args.seed)
    with open(out) as f:
        sample = json.load(f)
    print(f"wrote {out} ({len(sample)} examples)")
    print(json.dumps(dataset_stats(sample), indent=2))


if __name__ == "__main__":
    main()
