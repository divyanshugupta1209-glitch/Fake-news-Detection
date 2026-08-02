# backend/source_cred.py
import os
import pandas as pd
from urllib.parse import urlparse

CRED_PATH = os.path.join("data", "external", "source_credibility.csv")

if os.path.exists(CRED_PATH):
    try:
        _df = pd.read_csv(CRED_PATH)
        # expect columns: source (domain), score (0..1)
        _cred_map = {str(r['source']).lower(): float(r['score']) for _, r in _df.iterrows()}
    except Exception:
        _cred_map = {}
else:
    _cred_map = {}

def domain_from_url(url: str):
    try:
        d = urlparse(url).netloc.lower()
        # remove www
        if d.startswith("www."):
            d = d[4:]
        return d
    except:
        return ""

def get_source_score(url: str):
    """
    Return a score in 0..1. If domain known in CSV, use that.
    Otherwise, heuristic: big domains -> 0.8, medium -> 0.5, unknown -> 0.3
    """
    dom = domain_from_url(url)
    if not dom:
        return 0.4
    if dom in _cred_map:
        return max(0.0, min(1.0, _cred_map[dom]))
    # simple heuristics (you can expand)
    if dom.endswith(".gov") or dom.endswith(".edu"):
        return 0.95
    if ".nytimes." in dom or "bbc." in dom or "reuters." in dom or "cnn." in dom:
        return 0.9
    if len(dom) < 20:
        return 0.6
    return 0.3
