import random
import re


# ------------------------------------------------------------
# Linguistic Pattern Analyzer (offline)
# ------------------------------------------------------------
def analyze_language(text):
    """
    Very light-weight linguistic heuristic engine
    Scans for suspicious linguistic traits common in fake news.
    """
    text_lower = text.lower()

    sensational = ["shocking", "unbelievable", "explosive", "leaked", "miracle"]
    emotional = ["fear", "panic", "anger", "evil", "disaster", "outrage"]
    contradictions = [r"\bbut\b", r"\bhowever\b", r"\balthough\b"]
    exaggeration = ["everyone knows", "guaranteed", "no doubt", "100% true"]

    score = 0
    flags = []

    if any(w in text_lower for w in sensational):
        flags.append("sensational tone")
        score += 0.12

    if any(w in text_lower for w in emotional):
        flags.append("emotion-heavy wording")
        score += 0.12

    if any(re.search(p, text_lower) for p in contradictions):
        flags.append("internal contradictions")
        score += 0.08

    if any(w in text_lower for w in exaggeration):
        flags.append("exaggerated certainty")
        score += 0.10

    return score, flags



# ------------------------------------------------------------
# OFFLINE HUMAN JOURNALIST-TONE GENERATOR (V7)
# ------------------------------------------------------------
def generate_explanation(final_label, final_score, signals):
    """
    Highly human-like journalist-style reasoning.
    Fully offline — relies only on heuristics + model probabilities you already produce.
    """

    claim = signals.get("claim_text", "")
    article = signals.get("article_text", "")
    model_label = signals["model_label"]
    model_conf = signals["model_conf"]
    scs_score = signals["scs_score"]
    scs_label = signals["scs_label"]
    image_score = signals.get("image_score")
    srct = signals["srct_vote"]
    rmc = signals["rmc_vote"]

    ling_score, ling_flags = analyze_language(article)

    # --------------------------------------------------------------------
    # 1. Opening Lines (More Authentic Journalist Feel)
    # --------------------------------------------------------------------
    opening_choices = [
        "After reviewing the available material, a clearer sense of the claim’s credibility begins to emerge.",
        "Once the statement is broken down and compared with contextual signals, a more grounded picture forms.",
        "A closer look at the narrative and supporting details reveals several telling indicators.",
        "Examining the claim through factual, linguistic, and contextual lenses offers useful clarity."
    ]
    opening = random.choice(opening_choices)

    # --------------------------------------------------------------------
    # 2. Text Model Interpretation (Human tone)
    # --------------------------------------------------------------------
    if model_label == "REAL":
        text_part = (
            f"The text analysis model interprets the claim as **likely genuine**, "
            f"with a confidence level of {model_conf:.2f}. "
            f"It aligns with patterns usually seen in verified or well-grounded reports."
        )
    else:
        text_part = (
            f"The automated text assessment raises doubts, "
            f"assigning a {model_conf:.2f} confidence towards the claim being **false**. "
            f"Certain structural and tonal elements resemble known misleading narratives."
        )

    # --------------------------------------------------------------------
    # 3. Source Credibility Score (SCS)
    # --------------------------------------------------------------------
    if scs_score >= 0.7:
        source_part = (
            f"The source involved here has a strong reliability background "
            f"({scs_label.lower()}), which generally supports authenticity."
        )
    elif scs_score >= 0.4:
        source_part = (
            f"The source shows a mixed reliability record ({scs_label.lower()}). "
            f"It neither strongly confirms nor invalidates the claim."
        )
    else:
        source_part = (
            f"The source credibility score is relatively low ({scs_label.lower()}), "
            f"which creates reasonable caution about the claim’s authenticity."
        )

    # --------------------------------------------------------------------
    # 4. Image Consistency Check
    # --------------------------------------------------------------------
    if image_score is None:
        image_part = (
            "Since no image was provided, the system relied entirely on textual "
            "patterns and source credibility cues."
        )
    else:
        if image_score >= 0.65:
            image_part = (
                f"The accompanying image appears consistent with the narrative "
                f"(visual match score: {image_score:.2f}), reducing concerns about manipulation."
            )
        elif image_score >= 0.45:
            image_part = (
                f"The image loosely matches the described event "
                f"(score: {image_score:.2f}), providing only moderate support."
            )
        else:
            image_part = (
                f"The image aligns poorly with the written claim "
                f"(score: {image_score:.2f}), suggesting possible misplacement or misleading context."
            )

    # --------------------------------------------------------------------
    # 5. SRCT (Reverse Truth) & RMC (Historical Memory Check)
    # --------------------------------------------------------------------
    contrast_part = ""

    if srct == "FAKE":
        contrast_part += (
            " A reverse-context comparison contradicts the claim, "
            "which often happens when a narrative has been reshaped from its original facts."
        )

    if rmc == "FAKE":
        contrast_part += (
            " A previously recorded version of a similar claim was marked false, "
            "adding further skepticism."
        )
    elif rmc == "REAL":
        contrast_part += (
            " A historically similar claim was once validated, "
            "slightly reinforcing its credibility."
        )

    # --------------------------------------------------------------------
    # 6. Linguistic Flags (more realistic)
    # --------------------------------------------------------------------
    if ling_flags:
        linguistic_part = (
            " The writing style contains subtle indicators—such as "
            + ", ".join(ling_flags)
            + "—which commonly appear in disputed or emotionally-charged stories."
        )
    else:
        linguistic_part = (
            " The language used is straightforward and lacks the typical signals of manipulation."
        )

    # --------------------------------------------------------------------
    # 7. Final Verdict (More Human Narrative)
    # --------------------------------------------------------------------
    if final_label == "REAL":
        verdict_part = (
            "Bringing all these observations together, the overall evidence leans toward "
            "the claim being **authentic**. While no method is perfect, the indicators "
            "align more with credible reporting than fabrication."
        )
    elif final_label == "FAKE":
        verdict_part = (
            "Combining the evidence, the balance of signals points toward the claim being **false**. "
            "The inconsistencies and weak credibility elements outweigh the supporting ones."
        )
    else:
        verdict_part = (
            "Even after weighing all factors, the system cannot confidently label the claim "
            "as real or fake. The mixed signals place it in an **uncertain** category, "
            "suggesting more verification is needed."
        )

    # --------------------------------------------------------------------
    # FINAL COMBINED EXPLANATION
    # --------------------------------------------------------------------
    final_explanation = (
        opening + " " +
        text_part + " " +
        source_part + " " +
        image_part + " " +
        contrast_part +
        linguistic_part + " " +
        verdict_part
    )

    return final_explanation.strip()
