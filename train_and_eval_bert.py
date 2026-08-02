# train_and_eval_bert.py
import os
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from imblearn.over_sampling import RandomOverSampler  # type: ignore # pip install imbalanced-learn

# -------------------------
# Configurations
# -------------------------
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 2e-5
MODEL_SAVE_DIR = "models/bert_fake_news_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# Dataset Class
# -------------------------
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=MAX_LEN,
            return_tensors='pt'
        )
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        return input_ids, attention_mask, torch.tensor(label)

# -------------------------
# Load & Prepare Data
# -------------------------
data_files = [
    "data/raw/gossipcop_fake.csv",
    "data/raw/gossipcop_real.csv",
    "data/raw/politifact_fake.csv",
    "data/raw/politifact_real.csv"
]

dfs = []
for file in data_files:
    df = pd.read_csv(file)
    # Ensure 'content' or fallback to 'title'
    text_col = 'content' if 'content' in df.columns else 'title'
    df = df[[text_col]]
    df['label'] = 0 if 'fake' in file else 1
    df.rename(columns={text_col: 'text'}, inplace=True)
    dfs.append(df)

full_df = pd.concat(dfs, ignore_index=True)

# -------------------------
# Oversample to balance classes
# -------------------------
ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(full_df[['text']], full_df['label'])
full_df_balanced = pd.DataFrame({'text': X_res['text'], 'label': y_res})

# -------------------------
# Tokenizer & Dataset
# -------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
dataset = NewsDataset(full_df_balanced['text'].tolist(), full_df_balanced['label'].tolist(), tokenizer)

# Train/Test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# -------------------------
# Model
# -------------------------
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.to(DEVICE)

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()

# -------------------------
# Training Loop
# -------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for input_ids, attention_mask, labels in train_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Training Loss: {total_loss/len(train_loader):.4f}")

# -------------------------
# Evaluation
# -------------------------
model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for input_ids, attention_mask, labels in test_loader:
        input_ids, attention_mask, labels = input_ids.to(DEVICE), attention_mask.to(DEVICE), labels.to(DEVICE)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
print(f"\n📊 Accuracy: {acc:.4f}\n")
print(classification_report(all_labels, all_preds, target_names=["FAKE", "REAL"]))

# -------------------------
# Save Model
# -------------------------
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
model.save_pretrained(MODEL_SAVE_DIR)
tokenizer.save_pretrained(MODEL_SAVE_DIR)
print(f"\n✅ Model & tokenizer saved to {MODEL_SAVE_DIR}")
