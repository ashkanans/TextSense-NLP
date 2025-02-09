import torch.nn as nn

from torch.utils.data import Dataset


class TextDataset(Dataset):
    """Custom Dataset for text classification with padding support."""

    def __init__(self, data, vocab, tokenizer):
        self.data = data
        self.vocab = vocab
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        label = sample["label"]
        tokens = self.tokenizer(text)
        input_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]
        return torch.tensor(input_ids, dtype=torch.long), torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """Collate function for DataLoader to pad sequences."""
    texts, labels = zip(*batch)  # Unzip batch into separate lists
    texts_padded = pad_sequence(texts, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)  # Stack labels into a tensor
    return texts_padded, labels


class OptimizedLSTMClassifier(nn.Module):
    """Optimized BiLSTM-based text classifier with dropout & batch normalization."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=2, dropout=0.3):
        super(OptimizedLSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # BiLSTM with multiple layers
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True, dropout=dropout)

        # Batch Normalization
        self.batch_norm = nn.BatchNorm1d(hidden_dim * 2)

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # Multiply by 2 for bidirectional LSTM

        # Activation function
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)

        # Concatenate forward and backward hidden states
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)

        # Apply batch normalization
        hidden = self.batch_norm(hidden)

        output = self.fc(hidden)
        return self.log_softmax(output)


class LSTMClassifier(nn.Module):
    """LSTM-based text classifier."""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = hidden[-1]  # Use the last layer's hidden state
        output = self.fc(hidden)
        return self.softmax(output)


import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class ParamLSTM(nn.Module):
    """Generalized Parameterized LSTM Model"""

    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1,
                 bidirectional=False, dropout=0.3):
        super(ParamLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1) if self.lstm.bidirectional else hidden[-1]
        output = self.fc(hidden)
        return self.log_softmax(output)
