# backend/model_utils.py
import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from typing import Tuple
import os

MODEL_DIR = os.path.join("models", "bert_fake_news_model")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load tokenizer + classifier + bert (for embeddings)
_tokenizer = None
_classifier = None
_embedder = None

def load_models():
    global _tokenizer, _classifier, _embedder
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    if _classifier is None:
        _classifier = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)
        _classifier.eval()
    if _embedder is None:
        _embedder = BertModel.from_pretrained("bert-base-uncased").to(DEVICE)
        _embedder.eval()
    return _tokenizer, _classifier, _embedder

def predict_label_and_confidence(text: str) -> Tuple[str, float]:
    """Return ('REAL'|'FAKE', confidence 0..1)"""
    tokenizer, clf, _ = load_models()
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        outputs = clf(**encoding)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = int(torch.argmax(probs, dim=1).item())
        conf = float(probs[0][pred].cpu().item())
    # label mapping: model trained FAKE=0, REAL=1
    return ("REAL" if pred == 1 else "FAKE"), conf

def get_embedding(text: str):
    """Return pooled embedding vector (cpu numpy)"""
    tokenizer, _, embedder = load_models()
    encoding = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=256).to(DEVICE)
    with torch.no_grad():
        out = embedder(**encoding)
        pooled = out.pooler_output  # shape (1, hidden)
    return pooled.cpu().numpy()[0]
