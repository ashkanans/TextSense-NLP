import os

import torch

from src.model import LSTMClassifier
from src.train import evaluate_model


def load_trained_model(model_path, vocab_size, embed_dim, hidden_dim, output_dim, device='cpu'):
    """Load a trained LSTM model from a saved checkpoint."""
    if not os.path.exists(model_path):
        print(f"Current working directory: {os.getcwd()}")
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    model = LSTMClassifier(vocab_size, embed_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def evaluate_on_test(model, test_loader, device='cpu'):
    """Evaluate the model on the test set."""
    accuracy = evaluate_model(model, test_loader, device)
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy
