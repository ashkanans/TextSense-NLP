import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.data_loader import load_dataset
from src.model import TextDataset, collate_fn, OptimizedLSTMClassifier
from src.train import train_model

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("data/HASPEEDE")

    # Split data into training and validation sets
    train_data, val_data = train_test_split(dataset["train-taskA.jsonl"], test_size=0.2, random_state=42)

    # Create vocabulary from training data
    vocab = {word: idx for idx, word in
             enumerate(set(word for sample in dataset["train-taskA.jsonl"] for word in sample["text"].split()))}
    vocab["<UNK>"] = len(vocab)

    # Create dataset objects
    train_dataset = TextDataset(train_data, vocab, lambda x: x.split())
    val_dataset = TextDataset(val_data, vocab, lambda x: x.split())

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize and train the model
    # model = LSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=128, output_dim=2)
    model = OptimizedLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=256, output_dim=2)

    train_model(model, train_loader, val_loader)

    # Save trained model
    torch.save(model.state_dict(), "model.pth")
    print("Model training complete and saved as model.pth")
