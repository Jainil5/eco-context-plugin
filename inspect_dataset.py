import argparse
import json

from data import dataset_stats, load_split


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="s", choices=["s", "m", "oracle", "sample"])
    args = p.parse_args()
    if args.split == "sample":
        from pathlib import Path

        path = Path("dataset/longmemeval_sample.json")
        with open(path) as f:
            data = json.load(f)
    else:
        data = load_split(args.split)
    stats = dataset_stats(data)
    print(json.dumps(stats, indent=2))
    ex = data[0]
    print("\nexample keys:", list(ex.keys()))
    print("sample question:", ex["question"][:120])
    print("sample answer:", ex["answer"])
    print("sessions:", len(ex["haystack_sessions"]))


if __name__ == "__main__":
    main()
