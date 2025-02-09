import os
import json
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_loader import load_dataset
from src.model import ParamLSTM, TextDataset, collate_fn
from src.train import evaluate_model
from src.baselines import random_baseline, majority_class_baseline, stratified_baseline, evaluate_baseline, \
    word2vec_baseline


def load_trained_model(model_path, vocab_size, embed_dim, hidden_dim, output_dim, num_layers, bidirectional, dropout, device="cpu"):
    """Load a trained LSTM model from a saved checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    model = ParamLSTM(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    model_path = "models/lstm_model.pth"  # 📌 Set model path
    dataset = load_dataset("data/HASPEEDE")

    # 📌 Use the train set to build vocabulary
    train_data, val_data = train_test_split(dataset["train-taskA.jsonl"], test_size=0.2, random_state=42)

    # Create vocabulary from train set
    vocab = {word: idx for idx, word in
             enumerate(set(word for sample in train_data for word in sample["text"].split()))}
    vocab["<UNK>"] = len(vocab)  # Assign unknown word token

    # 📌 Load trained LSTM model
    model_config = {
        "hidden_dim": 64,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
        "embed_dim": 100,
        "output_dim": 2
    }
    model = load_trained_model(
        model_path, len(vocab), model_config["embed_dim"], model_config["hidden_dim"],
        model_config["output_dim"], model_config["num_layers"], model_config["bidirectional"],
        model_config["dropout"]
    )

    # 📌 Step 1: Evaluate LSTM Model on Test Sets
    test_results = {}
    for test_set_name in ["test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]:
        test_data = dataset[test_set_name]
        test_dataset = TextDataset(test_data, vocab, lambda x: x.split())
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

        test_results[test_set_name] = evaluate_model(model, test_loader)

    # 📌 Step 2: Evaluate Baselines on Test Sets
    print("\nEvaluating Baselines on Test Sets (News & Tweets)...\n")
    baseline_results = {}

    for test_set_name in ["test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]:
        test_data = dataset[test_set_name]

        baselines = {
            "Random Baseline": random_baseline(test_data),
            "Majority Baseline": majority_class_baseline(test_data),
            "Stratified Baseline": stratified_baseline(test_data),
            "Word2Vec Baseline": word2vec_baseline(train_data, test_data)
        }

        baseline_results[test_set_name] = {name: evaluate_baseline(preds, test_data) for name, preds in baselines.items()}

    print("✅ Baseline Evaluation Completed\n")

    # 📌 Step 3: Save Evaluation Results to JSON
    results = {
        "model_path": model_path,
        "model_config": model_config,
        "lstm_results": test_results,
        "baseline_results": baseline_results
    }

    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    json_path = os.path.join(results_dir, "evaluation_results.json")

    with open(json_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"✅ Evaluation complete. Results saved at {json_path}")
