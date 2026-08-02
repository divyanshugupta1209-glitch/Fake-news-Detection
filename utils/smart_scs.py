# utils/smart_scs.py
import os
import csv
import tldextract

# ---------------- CSV PATH ----------------
CRED_CSV = os.path.join("data", "processed", "external", "source_credibility.csv")

# ---------------- LOAD CSV ----------------
_domain_scores = {}
if os.path.exists(CRED_CSV):
    try:
        with open(CRED_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                dom = r.get("domain", "").strip().lower()
                score = float(r.get("score", 0.55))
                _domain_scores[dom] = score
    except Exception:
        pass


# ---------------- HELPER: EXTRACT DOMAIN ----------------
def extract_domain(url: str) -> str:
    try:
        ext = tldextract.extract(url)
        domain = ext.domain + "." + ext.suffix
        return domain.lower()
    except:
        return ""


# ---------------- HELPER: SCORE TIERS ----------------
def _tiered_scoring(domain: str) -> (float, str):
    """
    Returns (score, label) based on tier rules
    """
    # Trusted Tier
    trusted = ["bbc", "reuters", "cnn", "guardian", "nytimes", "apnews", "forbes"]
    if any(k in domain for k in trusted):
        return 0.90, "REAL"

    # Govt / Edu Tier
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95, "REAL"

    # Suspicious / Low credibility
    suspicious = ["blogspot", "wordpress", "click", "buzz", "viral", "info"]
    if any(k in domain for k in suspicious):
        return 0.35, "FAKE"

    # Unknown / neutral domains
    return 0.55, "NEUTRAL"


# ---------------- MAIN FUNCTION ----------------
def compute_scs_score(url: str) -> (float, str):
    """
    Computes Source Credibility Score (0–1) for a URL
    Returns: score, label
    """
    domain = extract_domain(url)

    # 1) Empty URL → neutral
    if not domain:
        return 0.55, "NEUTRAL"

    # 2) CSV-based domain score
    if domain in _domain_scores:
        s = _domain_scores[domain]
        label = "REAL" if s >= 0.60 else "FAKE"
        return s, label

    # 3) Apply tiered heuristics
    return _tiered_scoring(domain)
