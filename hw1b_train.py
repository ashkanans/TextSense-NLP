import os
import torch
from datetime import datetime
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.model import ParamLSTM, TextDataset, collate_fn
from src.train import train_model

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("data/HASPEEDE")

    # Split train set into train (80%) and validation (20%)
    train_data, val_data = train_test_split(dataset["train-taskA.jsonl"], test_size=0.2, random_state=42)

    # Create vocabulary from train set
    vocab = {word: idx for idx, word in
             enumerate(set(word for sample in train_data for word in sample["text"].split()))}
    vocab["<UNK>"] = len(vocab)  # Assign unknown word token

    # Create datasets and DataLoaders
    train_dataset = TextDataset(train_data, vocab, lambda x: x.split())
    val_dataset = TextDataset(val_data, vocab, lambda x: x.split())

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # 📌 Model Configuration
    model_config = {
        "hidden_dim": 64,
        "num_layers": 2,
        "bidirectional": True,
        "dropout": 0.3,
        "lr": 0.0005,
        "epochs": 5
    }

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

    # Train model on train set
    train_model(model, train_loader, val_loader, epochs=model_config["epochs"], lr=model_config["lr"])

    # Save trained model
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "lstm_model.pth")
    torch.save(model.state_dict(), model_path)

    print(f"✅ Training complete. Model saved at {model_path}")
