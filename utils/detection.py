# utils/detection.py
import os
os.environ["USE_TF"] = "0"

import csv
import torch
import tldextract
from dotenv import load_dotenv
import requests
import json
import re

load_dotenv()

from utils import db
from utils import image_expert
from utils import image_ocr
from utils.explanation_generator import generate_explanation
from transformers import BertTokenizer, BertForSequenceClassification
from utils import ai_image_detector


# ==========================================================
# 1. LOAD BERT MODEL
# ==========================================================
MODEL_DIR = "DivyanshuGupta/neo-refute-bert"
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_tokenizer = None
_model = None


def _load_model():
    global _tokenizer, _model
    if _tokenizer is None:
        _tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
    if _model is None:
        _model = BertForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
        _model.eval()
    return _tokenizer, _model


# ==========================================================
# 2. PRIMARY BERT CLASSIFIER
# ==========================================================
def model_predict(text):
    tokenizer, model = _load_model()
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(_device)

    with torch.no_grad():
        out = model(**enc)
        probs = torch.softmax(out.logits, dim=1)[0].cpu().numpy()

    fake_p, real_p = float(probs[0]), float(probs[1])
    label = "REAL" if real_p > fake_p else "FAKE"
    conf  = max(fake_p, real_p)

    return label, conf, fake_p, real_p


# ==========================================================
# 3. SRCT — Semantic Reverse Contrast Test
# ==========================================================
def srct_process(text):
    counter = "It is false that " + text
    l1, c1, _, _ = model_predict(text)
    l2, c2, _, _ = model_predict(counter)
    if l1 != l2 and max(c1, c2) >= 0.82:
        return "FAKE"
    return l1


# ==========================================================
# 4. RMC — Retrieval Memory Check
# ==========================================================
def rmc_process(claim_text):
    sim = db.find_similar_claim(claim_text, threshold=0.88)
    return sim["result"] if sim else "UNKNOWN"


# ==========================================================
# 5. SOURCE CREDIBILITY
# ==========================================================
CRED_CSV = os.path.join("data", "processed", "external", "source_credibility.csv")
_domain_scores = {}

if os.path.exists(CRED_CSV):
    try:
        with open(CRED_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames and "domain" in reader.fieldnames and "score" in reader.fieldnames:
                for r in reader:
                    try:
                        dom   = r["domain"].strip().lower()
                        score = float(r["score"])
                        _domain_scores[dom] = score
                    except (ValueError, KeyError):
                        continue
    except Exception as e:
        print("Warning loading CSV:", e)


def extract_domain(url):
    try:
        ext = tldextract.extract(url)
        return (ext.domain + "." + ext.suffix).lower()
    except:
        return ""


def scs_process(url):
    domain = extract_domain(url)
    if not domain:
        return 0.55, "NEUTRAL"

    if domain in _domain_scores:
        s     = _domain_scores[domain]
        label = "REAL" if s >= 0.6 else "FAKE"
        return s, label

    trusted = ["bbc", "reuters", "cnn", "guardian", "nytimes", "apnews", "forbes"]
    if any(k in domain for k in trusted):
        return 0.90, "REAL"

    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95, "REAL"

    bad = ["blogspot", "wordpress", "click", "buzz", "viral"]
    if any(k in domain for k in bad):
        return 0.35, "FAKE"

    return 0.55, "NEUTRAL"


# ==========================================================
# 6. UNCERTAINTY GATE
# ==========================================================
def uga_process(conf, threshold=0.60):
    return "UNCERTAIN" if conf < threshold else "CONFIDENT"


# ==========================================================
# 7. HF / OpenRouter AI Layer  ← FIXED
# ==========================================================
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL            = "https://openrouter.ai/api/v1/chat/completions"
MODEL              = "openai/gpt-4o-mini"

HEADERS = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type":  "application/json"
}


def query_hf_model(news_text: str) -> dict:
    """
    Calls OpenRouter API and safely parses the response.
    Returns dict with keys: label, score, explanation
    """
    if not OPENROUTER_API_KEY:
        return {
            "label":       "UNCERTAIN",
            "score":        0.5,
            "explanation": "Missing OPENROUTER_API_KEY in .env"
        }

    prompt = f"""You are a professional fact-checking AI.

Analyze the claim carefully and classify it as REAL, FAKE, or UNCERTAIN.

Use these rules:

REAL: The claim is supported by reliable and verifiable evidence or well-established facts.

FAKE: The claim is clearly false, misleading, or contradicted by reliable evidence.

UNCERTAIN: The claim cannot currently be verified, lacks sufficient reliable evidence, is an unconfirmed report, opinion, prediction, future event, or speculation.

IMPORTANT:
- Do not force a future prediction or opinion into REAL or FAKE.
- If there is not enough evidence to confidently classify the claim as REAL or FAKE, choose UNCERTAIN.
- Be careful with claims about future events, predictions, rumors, and unverified discoveries.

Return output in EXACTLY this format:
Label: REAL or FAKE or UNCERTAIN
Score: number between 0 and 1
Explanation: brief reasoning

News:
{news_text}
"""

    payload = {
        "model":       MODEL,
        "messages":    [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "max_tokens":  150
    }

    try:
        response = requests.post(
            API_URL,
            headers=HEADERS,
            data=json.dumps(payload),
            timeout=30
        )

        # ── Safe response parsing ──────────────────────────
        if response.status_code != 200:
            print(f"⚠️ API HTTP error: {response.status_code}")
            print(f"   Response: {response.text[:200]}")
            return {
                "label":       "UNCERTAIN",
                "score":        0.5,
                "explanation": f"API error {response.status_code}: {response.text[:100]}"
            }

        data = response.json()

        # Check for API-level errors
        if "error" in data:
            err_msg = data["error"].get("message", "Unknown API error")
            print(f"⚠️ API error: {err_msg}")
            return {
                "label":       "UNCERTAIN",
                "score":        0.5,
                "explanation": f"API error: {err_msg}"
            }

        # Check choices exist
        if "choices" not in data or not data["choices"]:
            print(f"⚠️ No choices in response: {data}")
            return {
                "label":       "UNCERTAIN",
                "score":        0.5,
                "explanation": "API returned empty response"
            }

        # Extract raw text
        raw = data["choices"][0]["message"]["content"].strip()
        print(f"✅ HF API response received ({len(raw)} chars)")

        raw_lower = raw.lower()

        # Parse label
        if "label: real" in raw_lower or raw_lower.startswith("real"):
            label = "REAL"
        elif "label: fake" in raw_lower or raw_lower.startswith("fake"):
            label = "FAKE"
        elif "real" in raw_lower and "fake" not in raw_lower:
            label = "REAL"
        elif "fake" in raw_lower:
            label = "FAKE"
        else:
            label = "UNCERTAIN"

        # Parse score
        score_match = re.search(r"score[:\s]+([01]\.\d+|0|1)", raw_lower)
        if score_match:
            score = float(score_match.group(1))
        else:
            # Fallback: find any decimal between 0-1
            any_match = re.search(r"\b(0\.\d+|1\.0+)\b", raw_lower)
            score = float(any_match.group(0)) if any_match else (
                0.85 if label == "REAL" else
                0.15 if label == "FAKE" else
                0.5
            )

        # Clamp score
        score = max(0.0, min(1.0, score))

        print(f"   Label: {label} | Score: {score:.3f}")

        return {
            "label":       label,
            "score":        score,
            "explanation": raw
        }

    except requests.exceptions.Timeout:
        print("⚠️ API request timed out")
        return {
            "label":       "UNCERTAIN",
            "score":        0.5,
            "explanation": "Request timed out. Please try again."
        }
    except requests.exceptions.ConnectionError:
        print("⚠️ Connection error - check internet")
        return {
            "label":       "UNCERTAIN",
            "score":        0.5,
            "explanation": "Connection error. Check your internet connection."
        }
    except Exception as e:
        print(f"⚠️ Unexpected error: {e}")
        return {
            "label":       "UNCERTAIN",
            "score":        0.5,
            "explanation": f"Error: {str(e)}"
        }


# ==========================================================
# 8. FUSION: HF + IMAGE (CLIP + OCR + AI Detection)
# ==========================================================
def fuse_predictions(
    claim_text,
    article_text,
    url,
    image=None,
    hf_result=None,
    uga_threshold=0.60
):
    """
    Fully multimodal fusion:
    - HF text is primary (70%)
    - Image (CLIP + OCR + AI Detection) is secondary (30%)
    - If no image: use HF only
    """

    # ── 1. HF Text Layer ──────────────────────────────────
    if hf_result is None:
        hf_result = query_hf_model(article_text)

    hf_label       = str(hf_result.get("label", "UNCERTAIN")).upper()
    hf_score       = float(hf_result.get("score", 0.5))
    hf_explanation = hf_result.get("explanation", "")

    # ── 2. Image Analysis ─────────────────────────────────
    image_score        = None
    ocr_result         = None
    ai_detection_result = None

    if image is not None:
        try:
            # 2a. CLIP Consistency (50% of image weight)
            consistency_score = image_expert.image_expert_consistency(image, claim_text)
            print(f"   ├─ CLIP Consistency:  {consistency_score:.3f}")

            # 2b. OCR Text Match (30% of image weight)
            ocr_result = image_ocr.analyze_image_text(image, claim_text)
            ocr_score  = ocr_result["score"]
            print(f"   ├─ OCR Text Match:    {ocr_score:.3f}")

            # 2c. AI Deepfake Detection (20% of image weight)
            ai_detection_result = ai_image_detector.detect_ai_generated(image)
            ai_score            = ai_detection_result["score"]
            print(f"   ├─ AI Detection:      {ai_score:.3f}")

            # Combined image score
            image_score = (
                0.50 * consistency_score +
                0.30 * ocr_score +
                0.20 * ai_score
            )
            print(f"   └─ Combined Image Score: {image_score:.3f}")

        except Exception as e:
            print(f"⚠️ Image analysis failed: {e}")
            image_score         = 0.5
            ocr_result          = None
            ai_detection_result = None

    # ── 3. Supporting Modules ─────────────────────────────
    model_label, model_conf, fake_p, real_p = model_predict(article_text)
    srct_vote            = srct_process(article_text)
    rmc_vote             = rmc_process(claim_text)
    scs_score, scs_label = scs_process(url)
    uga_vote             = uga_process(model_conf, uga_threshold)

    # ── 4. Fusion ─────────────────────────────────────────
    if image_score is not None:
        combined_score = 0.7 * hf_score + 0.3 * image_score
    else:
        combined_score = hf_score

    combined_score = max(0.0, min(1.0, combined_score))

    # Respect UNCERTAIN claims identified by the AI
    if hf_label == "UNCERTAIN":
        final_label = "UNCERTAIN"
    elif combined_score >= 0.6:
        final_label = "REAL"
    elif combined_score <= 0.4:
        final_label = "FAKE"
    else:
        final_label = "UNCERTAIN"

    # ── 5. Explanation ────────────────────────────────────
    explanation = generate_explanation(
        final_label,
        combined_score,
        {
            "hf_score":            hf_score,
            "hf_label":            hf_label,
            "hf_explanation":      hf_explanation,
            "image_score":         image_score,
            "model_label":         model_label,
            "model_conf":          model_conf,
            "fake_prob":           fake_p,
            "real_prob":           real_p,
            "srct_vote":           srct_vote,
            "rmc_vote":            rmc_vote,
            "scs_score":           scs_score,
            "scs_label":           scs_label,
            "uga_vote":            uga_vote,
            "ocr_result":          ocr_result,
            "ai_detection_result": ai_detection_result,
        }
    )

    # ── 6. Return full details ────────────────────────────
    return final_label, combined_score, {
        # HF
        "hf_score":            hf_score,
        "hf_label":            hf_label,
        "hf_explanation":      hf_explanation,
        # Image
        "image_score":         image_score,
        "ocr_result":          ocr_result,
        "ai_detection_result": ai_detection_result,
        # Supporting modules
        "model_label":         model_label,
        "model_conf":          model_conf,
        "fake_prob":           fake_p,
        "real_prob":           real_p,
        "srct_vote":           srct_vote,
        "rmc_vote":            rmc_vote,
        "scs_score":           scs_score,
        "scs_label":           scs_label,
        "uga_vote":            uga_vote,
        # Final
        "combined_score":      combined_score,
        "final_label":         final_label,
        "explanation":         explanation,
    }
