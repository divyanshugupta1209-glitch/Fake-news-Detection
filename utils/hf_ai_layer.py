# utils/hf_ai_layer.py

import os
import requests
import re
from dotenv import load_dotenv
import json

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

MODEL = "openai/gpt-5.2"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json"
}

# -----------------------------
# Helpers
# -----------------------------

def extract_label(text: str) -> str:
    text_low = text.lower()
    if "label: real" in text_low:
        return "real"
    if "label: fake" in text_low:
        return "fake"

    fake_hits = ["appears to be fake", "likely fake", "misleading", "false claim"]
    real_hits = ["appears to be real", "likely real", "mostly real", "real news"]

    if any(x in text_low for x in fake_hits):
        return "fake"
    if any(x in text_low for x in real_hits):
        return "real"

    if "fake" in text_low: return "fake"
    if "real" in text_low: return "real"

    return "unsure"

def extract_confidence(text: str) -> float:
    match = re.search(r"\b(0\.\d{1,4}|1\.0+)\b", text)
    if match: return float(match.group(1))
    match = re.search(r"\b(\d{1,3})%\b", text)
    if match: return float(match.group(1))/100
    return 0.5

# -----------------------------
# Main function
# -----------------------------

def query_hf_model(news_text: str) -> dict:
    """
    Query DeepSeek via OpenRouter for real-time fake news detection.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "Missing OPENROUTER_API_KEY"}

    prompt = (
        "You are a fact-checking AI.\n"
        "Classify the news below as REAL or FAKE.\n"
        "Output format:\n"
        "Label: REAL or FAKE\n"
        "Score: 0-1 confidence\n"
        "Explanation: short reasoning\n\n"
        f"News:\n{news_text}"
    )

    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens": 300
    }

    try:
        response = requests.post(API_URL, headers=HEADERS, data=json.dumps(payload), timeout=30)
        if response.status_code != 200:
            return {"label":"unsure","score":0.5,
                    "explanation": f"API status {response.status_code}: {response.text}"}

        data = response.json()
        raw = data["choices"][0]["message"]["content"].strip()
        if not raw:
            return {"label":"unsure","score":0.5,"explanation":"Empty response from API"}

        label = extract_label(raw)
        score = extract_confidence(raw)

        return {"label": label, "score": float(score), "explanation": raw}

    except Exception as e:
        return {"label":"unsure","score":0.5,"explanation": f"Exception: {str(e)}"}
