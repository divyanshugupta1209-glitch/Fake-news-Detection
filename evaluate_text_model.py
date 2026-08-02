import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from collections import Counter

# -------------------------
# Dataset Class
# -------------------------
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=100):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len

        # Build vocabulary if not provided
        if vocab is None:
            all_tokens = [tok for text in texts for tok in text.split()]
            self.vocab = {w: i+1 for i, (w, _) in enumerate(Counter(all_tokens).most_common())}
        else:
            self.vocab = vocab

        self.vocab_size = len(self.vocab) + 1

    def encode(self, text):
        tokens = text.split()
        ids = [self.vocab.get(tok, 0) for tok in tokens[:self.max_len]]
        if len(ids) < self.max_len:
            ids += [0] * (self.max_len - len(ids))  # pad to max_len
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.encode(self.texts[idx]), torch.tensor(self.labels[idx], dtype=torch.long)


# -------------------------
# Model Class
# -------------------------
class TextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, hidden_dim=768, num_classes=2):
        super(TextModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (h, _) = self.lstm(x)
        return self.fc(h[-1])


# -------------------------
# Main Evaluation
# -------------------------
DATA_PATH = "data/processed/text_test.csv"
MODEL_PATH = "models/saved_model/text_model.pt"

# Load test data
df = pd.read_csv(DATA_PATH)

# Determine which column has text
possible_text_cols = ["content", "text", "title", "news"]
text_col = next((c for c in possible_text_cols if c in df.columns), None)
if text_col is None:
    raise ValueError(f"No text column found in {df.columns.tolist()}")

print(f"✅ Using column '{text_col}' for evaluation")

texts = df[text_col].astype(str).tolist()
labels = df["label"].tolist()

# Convert string labels to 0/1
labels = [1 if str(l).upper() == "REAL" else 0 for l in labels]

# Dataset and DataLoader
dataset = TextDataset(texts, labels, max_len=100)
loader = DataLoader(dataset, batch_size=32, shuffle=False)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextModel(vocab_size=dataset.vocab_size)
state_dict = torch.load(MODEL_PATH, map_location=device)

# Handle possible key mismatch
new_state_dict = {}
for k, v in state_dict.items():
    new_state_dict[k.replace("classifier", "fc")] = v

model.load_state_dict(new_state_dict, strict=False)
model.to(device)
model.eval()

# Evaluation
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, batch_labels in loader:
        inputs, batch_labels = inputs.to(device), batch_labels.to(device)
        outputs = model(inputs)
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_labels.cpu().numpy())

# Metrics
acc = accuracy_score(all_labels, all_preds)
print(f"\n📊 Accuracy: {acc:.4f}\n")
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))
