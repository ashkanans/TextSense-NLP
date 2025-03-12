import os
import json
from typing import List, Dict


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from a JSONL file."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def load_dataset(data_dir: str = "data/HASPEEDE") -> Dict[str, List[Dict]]:
    """
    Load train and test datasets from JSONL files.

    Returns:
        dataset (dict): A dictionary containing:
            - "train-taskA.jsonl": Training dataset.
            - "test-news-taskA.jsonl": News test dataset.
            - "test-tweets-taskA.jsonl": Tweets test dataset.
    """
    dataset = {}
    files = ["train-taskA.jsonl", "test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]

    for file in files:
        file_path = os.path.join(data_dir, file)
        if os.path.exists(file_path):
            dataset[file] = load_jsonl(file_path)
        else:
            print(f"Warning: {file_path} not found.")

    return dataset


def get_label_distribution(dataset: List[Dict]) -> Dict[str, int]:
    """Compute label distribution in the dataset."""
    label_counts = {}
    for item in dataset:
        label = item["choices"][item["label"]]  # Convert label index to label name
        label_counts[label] = label_counts.get(label, 0) + 1
    return label_counts


if __name__ == "__main__":
    # Load dataset and display some statistics
    data = load_dataset()
    for key, value in data.items():
        print(f"{key}: {len(value)} samples")
        print("Label Distribution:", get_label_distribution(value))
