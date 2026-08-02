import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torchvision import models, transforms
from PIL import Image
import requests
from io import BytesIO
from app.modules import srct, rmc, scs, uga
from app.utils import preprocess_text, fuse_predictions

# ---------- TEXT MODEL ----------
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
text_model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
text_model.eval()

# ---------- IMAGE MODEL ----------
image_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
num_features = image_model.fc.in_features
image_model.fc = nn.Linear(num_features, 2)  # Binary FAKE/REAL classifier
image_model.eval()

image_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# ---------- TEXT FEATURES ----------
def extract_text_features(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    return inputs

# ---------- IMAGE FEATURES ----------
def extract_image_features(image_url: str):
    try:
        response = requests.get(image_url)
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img_tensor = image_transform(img).unsqueeze(0)
        with torch.no_grad():
            outputs = image_model(img_tensor)
            pred = torch.argmax(outputs, dim=-1).item()
            return "REAL" if pred == 1 else "FAKE"
    except:
        return "UNCERTAIN"

# ---------- FINAL CLASSIFICATION ----------
def classify(text: str, image_url: str = None):
    clean_text = preprocess_text(text)
    text_inputs = extract_text_features(clean_text)

    # Text prediction
    with torch.no_grad():
        text_outputs = text_model(**text_inputs)
        logits = text_outputs.logits
        text_pred = "REAL" if torch.argmax(logits, dim=-1).item() == 1 else "FAKE"

    # Image prediction
    img_pred = "UNCERTAIN"
    if image_url:
        img_pred = extract_image_features(image_url)

    # Special modules
    srct_out = srct.process(clean_text)
    rmc_out = rmc.process(clean_text)
    scs_out = scs.process(clean_text)
    uga_out = uga.process(clean_text, text_pred)

    # Fusion
    final_pred = fuse_predictions(text_pred, srct_out, rmc_out, scs_out, uga_out, img_pred)
    return final_pred
