import json
import os
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_dataset(data_dir="../data/HASPEEDE"):
    """Load train and test datasets from JSONL files."""
    dataset = {}
    files = ["train-taskA.jsonl", "test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            dataset[file] = load_jsonl(file_path)
        else:
            print(f"Warning: {file_path} not found.")

    return dataset


def get_label_distribution(dataset: List[Dict]) -> Dict:
    """Compute label distribution in the dataset."""
    label_counts = {}
    for item in dataset:
        label = item["choices"][item["label"]]  # Convert label index to label name
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts
