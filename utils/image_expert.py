# utils/image_expert.py

import os
import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CLIP Model Setup (FREE - Runs Locally)
# ----------------------------
print("🔄 Loading CLIP model (one-time download)...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
print(f"✅ CLIP model loaded on {device}")

# ----------------------------
# Core: Image-Claim Consistency Check
# ----------------------------
def image_expert_consistency(image, claim_text):
    """
    Uses CLIP to measure semantic similarity between image and text.
    Returns: 0-1 float (higher = more consistent)
    
    How it works:
    - Compares the image against the original claim
    - Also compares against a negated version
    - Returns probability that image matches the claim
    """
    try:
        # Ensure image is RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create multiple text comparisons for better accuracy
        texts = [
            claim_text,  # Original claim
            f"This image shows: {claim_text}",  # Positive reinforcement
            f"This image does NOT show: {claim_text}",  # Negative version
            "A random unrelated image"  # Baseline noise
        ]

        # Process inputs
        inputs = processor(
            text=texts,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        # Get similarity scores
        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # Shape: [1, 4]
            probs = logits_per_image.softmax(dim=1)[0]  # Convert to probabilities

        # Calculate final score
        # High score if image matches claim (index 0 and 1)
        # Low score if image contradicts (index 2 and 3)
        positive_score = float((probs[0] + probs[1]) / 2)
        negative_score = float((probs[2] + probs[3]) / 2)
        
        # Final consistency score
        # If positive >> negative, score is high
        # If negative >> positive, score is low
        raw_score = positive_score / (positive_score + negative_score + 1e-6)
        
        # Normalize to 0-1 range with better calibration
        # CLIP tends to be conservative, so we adjust the scale
        if raw_score > 0.6:
            final_score = min(0.5 + (raw_score - 0.6) * 1.25, 1.0)
        elif raw_score < 0.4:
            final_score = max(raw_score * 0.8, 0.0)
        else:
            final_score = raw_score

        print(f"✅ Image-Text Consistency Score: {final_score:.3f}")
        print(f"   Positive match: {positive_score:.3f} | Negative match: {negative_score:.3f}")
        
        return float(final_score)

    except Exception as e:
        print(f"⚠️ CLIP Error: {str(e)}")
        return 0.5  # Neutral score on error


# ----------------------------
# Optional: Detailed Analysis Function
# ----------------------------
def analyze_image_detailed(image, claim_text, top_k=5):
    """
    Extended analysis with multiple candidate descriptions.
    Useful for debugging or detailed explanations.
    """
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Generate alternative interpretations
        candidates = [
            claim_text,
            f"A photo showing {claim_text.lower()}",
            f"An image of {claim_text.lower()}",
            f"A picture depicting {claim_text.lower()}",
            "A completely unrelated image",
            "A random photograph",
        ]

        inputs = processor(
            text=candidates,
            images=image,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = logits_per_image.softmax(dim=1)[0]

        # Get top matches
        top_probs, top_indices = torch.topk(probs, min(top_k, len(candidates)))
        
        results = []
        for prob, idx in zip(top_probs, top_indices):
            results.append({
                "text": candidates[idx],
                "probability": float(prob)
            })

        return results

    except Exception as e:
        print(f"⚠️ Detailed analysis error: {str(e)}")
        return []