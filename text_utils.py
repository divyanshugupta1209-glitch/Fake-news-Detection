import torch
import torch.nn as nn
from torch.utils.data import Dataset

# -----------------------------
# Dataset for Fake News Text
# -----------------------------
class FakeNewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Convert labels: "REAL" -> 1, "FAKE" -> 0
        if isinstance(label, str):
            label = 1 if label.upper() == "REAL" else 0

        # Tokenize
        tokens = self.tokenizer(text)
        if len(tokens) > self.max_len:
            tokens = tokens[:self.max_len]

        # Convert to tensor
        ids = torch.tensor(tokens, dtype=torch.long)
        label = torch.tensor(label, dtype=torch.long)

        return ids, label


# -----------------------------
# BERT-based Model (matches checkpoint)
# -----------------------------
class TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=768, num_classes=2):  # set hidden_dim=768
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])
