# utils/fusion_engine.py

def fuse_scores(text_label, text_conf, img_consistency, scs_score):
    """
    Final Fusion Engine for NEO-REFUTE
    ----------------------------------
    Inputs:
        text_label      : "REAL" or "FAKE" (from text expert)
        text_conf       : text model confidence (0–1)
        img_consistency : refined CLIP consistency score (0–1)
        scs_score       : Source Credibility Score (0–1)

    Fusion Logic:
        - Text model is the strongest signal               → 55%
        - Image–text consistency affects reliability       → 20%
        - Source Credibility Score stabilizes the result   → 25%
    """

    # If no image provided → use neutral value
    if img_consistency is None:
        img_consistency = 0.50

    # Weighted aggregation
    final_score = (
        0.55 * float(text_conf) +
        0.20 * float(img_consistency) +
        0.25 * float(scs_score)
    )

    # Final binary decision
    final_label = "REAL" if final_score >= 0.60 else "FAKE"

    return {
        "final_label": final_label,
        "final_score": round(final_score, 3),

        "text_label": text_label,
        "text_confidence": round(text_conf, 3),

        "image_consistency": round(img_consistency, 3),
        "scs_score": round(scs_score, 3)
    }
