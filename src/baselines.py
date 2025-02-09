import random
from collections import Counter
from typing import List, Dict


def random_baseline(dataset: List[Dict]) -> List[str]:
    """Randomly assigns a label from available choices for each sample."""
    predictions = [random.choice(sample["choices"]) for sample in dataset]
    return predictions


def majority_class_baseline(dataset: List[Dict]) -> List[str]:
    """Predicts the most frequent class in the training dataset."""
    label_counts = Counter(sample["choices"][sample["label"]] for sample in dataset)
    majority_label = label_counts.most_common(1)[0][0]
    return [majority_label] * len(dataset)


def stratified_baseline(dataset: List[Dict]) -> List[str]:
    """Predicts labels based on their distribution in the training dataset."""
    label_counts = Counter(sample["choices"][sample["label"]] for sample in dataset)
    total_samples = sum(label_counts.values())
    label_probs = {label: count / total_samples for label, count in label_counts.items()}

    predictions = [random.choices(list(label_probs.keys()), weights=label_probs.values())[0] for _ in dataset]
    return predictions
