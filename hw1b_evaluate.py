import torch
from torch.utils.data import DataLoader

from src.data_loader import load_dataset
from src.evaluate import evaluate_on_test
from src.model import TextDataset, OptimizedLSTMClassifier, collate_fn

if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("data/HASPEEDE")

    # Create vocabulary from training data
    vocab = {word: idx for idx, word in
             enumerate(set(word for sample in dataset["train-taskA.jsonl"] for word in sample["text"].split()))}
    vocab["<UNK>"] = len(vocab)

    # Load test data
    test_data = dataset["test-news-taskA.jsonl"] + dataset["test-tweets-taskA.jsonl"]
    test_dataset = TextDataset(test_data, vocab, lambda x: x.split())
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Load trained model
    model = OptimizedLSTMClassifier(vocab_size=len(vocab), embed_dim=100, hidden_dim=256, output_dim=2)
    model.load_state_dict(torch.load("model.pth"))
    model.eval()

    # Evaluate model on test set
    evaluate_on_test(model, test_loader)
    print("Evaluation complete.")
