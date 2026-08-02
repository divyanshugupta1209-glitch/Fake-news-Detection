# train_text_model.py
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm  # progress bar

# ---------- Dataset ----------
class TextDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        text = str(row['title']) + " " + str(row['content'])
        label = 1 if row['label'] == 'REAL' else 0

        tokens = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        input_ids = tokens['input_ids'].squeeze(0)
        attention_mask = tokens['attention_mask'].squeeze(0)

        return input_ids, attention_mask, label

# ---------- Model ----------
class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = AutoModel.from_pretrained("distilbert-base-uncased")
        self.fc = nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask):
        x = self.bert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        x = self.fc(x)
        return x

# ---------- Training ----------
dataset = TextDataset("data/processed/text_train.csv")
loader = DataLoader(dataset, batch_size=8, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TextModel().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):
    total_loss = 0
    progress_bar = tqdm(loader, desc=f"Epoch {epoch+1}", unit="batch")

    for input_ids, attention_mask, labels in progress_bar:
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    print(f"Epoch {epoch+1} completed. Avg Loss: {total_loss/len(loader):.4f}")

# Save model
import os
os.makedirs("models/saved_model", exist_ok=True)
torch.save(model.state_dict(), "models/saved_model/text_model.pt")
print("✅ Text model training complete. Saved to models/saved_model/text_model.pt")
