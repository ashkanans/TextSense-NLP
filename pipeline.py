import json
import os
from datetime import datetime
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.baselines import evaluate_baseline, random_baseline, majority_class_baseline, stratified_baseline
from src.data_loader import load_dataset
from src.model import ParamLSTM, TextDataset, collate_fn
from src.train import train_model, evaluate_model

if __name__ == "__main__":
    # Load datasets
    dataset = load_dataset("data/HASPEEDE")

    # Split train set into train (80%) and validation (20%)
    train_data, val_data = train_test_split(dataset["train-taskA.jsonl"], test_size=0.2, random_state=42)

    # Create vocabulary only from train set
    vocab = {word: idx for idx, word in
             enumerate(set(word for sample in train_data for word in sample["text"].split()))}
    vocab["<UNK>"] = len(vocab)  # Assign unknown word token

    # Create datasets and DataLoaders
    train_dataset = TextDataset(train_data, vocab, lambda x: x.split())
    val_dataset = TextDataset(val_data, vocab, lambda x: x.split())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 📌 Step 1: Evaluate Baselines on Test Sets (News & Tweets)
    print("\nEvaluating Baselines on Test Sets (News & Tweets)...\n")
    baseline_results = {}

    for test_set_name in ["test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]:
        test_data = dataset[test_set_name]

        baselines = {
            "Random Baseline": random_baseline(test_data),
            "Majority Baseline": majority_class_baseline(test_data),
            "Stratified Baseline": stratified_baseline(test_data),
        }

        # Evaluate each baseline
        baseline_results[test_set_name] = {name: evaluate_baseline(preds, test_data) for name, preds in baselines.items()}

    print("✅ Baseline Evaluation Completed\n")

    # 📌 Step 2: Define Model Configurations for Experiments
    model_configs = [
        {"hidden_dim": 64, "num_layers": 2, "bidirectional": True, "dropout": 0.3, "lr": 0.0005, "epochs": 20},
        {"hidden_dim": 32, "num_layers": 3, "bidirectional": True, "dropout": 0.3, "lr": 0.0005, "epochs": 20},
        {"hidden_dim": 64, "num_layers": 2, "bidirectional": True, "dropout": 0.3, "lr": 0.0004, "epochs": 15},
        {"hidden_dim": 32, "num_layers": 2, "bidirectional": True, "dropout": 0.2, "lr": 0.0005, "epochs": 15},
        {"hidden_dim": 64, "num_layers": 2, "bidirectional": True, "dropout": 0.3, "lr": 0.0005, "epochs": 20},
        {"hidden_dim": 32, "num_layers": 2, "bidirectional": True, "dropout": 0.3, "lr": 0.0003, "epochs": 15},
        {"hidden_dim": 64, "num_layers": 3, "bidirectional": True, "dropout": 0.25, "lr": 0.0005, "epochs": 15},
        {"hidden_dim": 32, "num_layers": 3, "bidirectional": True, "dropout": 0.3, "lr": 0.0003, "epochs": 15},
        {"hidden_dim": 64, "num_layers": 4, "bidirectional": True, "dropout": 0.3, "lr": 0.0005, "epochs": 15},
        {"hidden_dim": 32, "num_layers": 2, "bidirectional": True, "dropout": 0.35, "lr": 0.0005, "epochs": 15},
    ]

    # 📌 Step 3: Train & Evaluate LSTM Models
    for model_config in model_configs:
        # Create timestamped experiment directory
        experiment_dir = os.path.join("experiments", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
        os.makedirs(experiment_dir, exist_ok=True)

        # Initialize model
        model = ParamLSTM(
            vocab_size=len(vocab),
            embed_dim=100,
            hidden_dim=model_config["hidden_dim"],
            output_dim=2,
            num_layers=model_config["num_layers"],
            bidirectional=model_config["bidirectional"],
            dropout=model_config["dropout"]
        )

        # Train model on train set and validate on val set
        train_model(model, train_loader, val_loader, epochs=model_config["epochs"], lr=model_config["lr"])

        # Save trained model
        model_path = os.path.join(experiment_dir, "model.pth")
        torch.save(model.state_dict(), model_path)

        # 📌 Step 4: Evaluate LSTM Model on Test Sets
        for test_set_name in ["test-news-taskA.jsonl", "test-tweets-taskA.jsonl"]:
            test_data = dataset[test_set_name]

            # Handle OOV words by mapping unknown words to <UNK>
            test_dataset = TextDataset(test_data, vocab, lambda x: x.split())
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

            # Evaluate on test set
            test_metrics = evaluate_model(model, test_loader)

            # Save results to model configuration
            model_config[f"{test_set_name}_accuracy"] = test_metrics[0]
            model_config[f"{test_set_name}_precision"] = test_metrics[1]
            model_config[f"{test_set_name}_recall"] = test_metrics[2]
            model_config[f"{test_set_name}_f1"] = test_metrics[3]

        # 📌 Step 5: Save LSTM & Baseline Results into JSON File
        results = {
            "model_config": model_config,
            "baseline_results": baseline_results
        }

        json_path = os.path.join(experiment_dir, "model_info.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)

        print(f"✅ Training and testing complete. Results saved in {experiment_dir}")
