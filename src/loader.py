import os
import json
import pandas as pd
from datasets import load_dataset

class SWEBenchLoader:
    def __init__(self, dataset_name="princeton-nlp/SWE-bench_Lite", split="test", cache_dir="dataset/swebench_cl"):
        self.dataset_name = dataset_name
        self.split = split
        self.cache_dir = cache_dir
        self.dataset_path = os.path.join(cache_dir, f"{dataset_name.replace('/', '_')}_{split}.jsonl")
        os.makedirs(cache_dir, exist_ok=True)

    def download_dataset(self):
        """Downloads the dataset and saves it locally if not already present."""
        if not os.path.exists(self.dataset_path):
            print(f"Downloading {self.dataset_name} ({self.split})...")
            dataset = load_dataset(self.dataset_name, split=self.split)
            dataset.to_json(self.dataset_path)
            print(f"Saved dataset to {self.dataset_path}")
        else:
            print(f"Dataset already exists at {self.dataset_path}")

    def load_tasks(self, limit=3):
        """Loads a limited number of tasks from the local dataset."""
        self.download_dataset()
        df = pd.read_json(self.dataset_path, lines=True)
        if limit is not None:
            df = df.head(limit)
        return df.to_dict(orient="records")

if __name__ == "__main__":
    loader = SWEBenchLoader()
    tasks = loader.load_tasks(limit=1)
    print(f"Loaded {len(tasks)} tasks.")
    if tasks:
        print(f"First task repo: {tasks[0]['repo']}, base_commit: {tasks[0]['base_commit']}")
