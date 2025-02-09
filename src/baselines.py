import random
from collections import Counter
from typing import List, Dict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def random_baseline(dataset: List[Dict]) -> List[str]:
    """Randomly assigns a label from available choices for each sample."""
    return [random.choice(sample["choices"]) for sample in dataset]


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
    return [random.choices(list(label_probs.keys()), weights=label_probs.values())[0] for _ in dataset]


def evaluate_baseline(predictions: List[str], dataset: List[Dict]) -> Dict[str, float]:
    """Compute Accuracy, Precision, Recall, and F1-score for baseline predictions."""
    true_labels = [sample["choices"][sample["label"]] for sample in dataset]

    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro', zero_division=0)
    recall = recall_score(true_labels, predictions, average='macro', zero_division=0)
    f1 = f1_score(true_labels, predictions, average='macro')

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


def compare_with_baselines(dataset: List[Dict], lstm_results: Dict[str, float]):
    """Compare trained LSTM results with baselines."""
    print("\nEvaluating Baselines...")

    baselines = {
        "Random Baseline": random_baseline(dataset),
        "Majority Baseline": majority_class_baseline(dataset),
        "Stratified Baseline": stratified_baseline(dataset),
    }

    baseline_results = {name: evaluate_baseline(preds, dataset) for name, preds in baselines.items()}

    print("\n🚀 **Comparison of LSTM vs. Baselines** 🚀\n")
    print(f"{'Model':<20}{'Accuracy':<10}{'Precision':<10}{'Recall':<10}{'F1-Score':<10}")
    print("-" * 60)

    # Print Baseline Results
    for name, results in baseline_results.items():
        print(
            f"{name:<20}{results['accuracy']:<10.4f}{results['precision']:<10.4f}{results['recall']:<10.4f}{results['f1_score']:<10.4f}")

    # Print LSTM Results
    print(
        f"\nLSTM Model{'':<13}{lstm_results['accuracy']:<10.4f}{lstm_results['precision']:<10.4f}{lstm_results['recall']:<10.4f}{lstm_results['f1_score']:<10.4f}")

    return baseline_results
