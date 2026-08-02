import os
import requests
import re
from dotenv import load_dotenv

load_dotenv()

ZENSERP_API_KEY = os.getenv("ZENSERP_API_KEY")


# ------------------------------------------------
# SMART KEYWORD EXTRACTOR
# ------------------------------------------------
def extract_keywords(text, max_words=10):
    """
    Extract simplified keywords from the text for fallback search.
    Removes stopwords and keeps unique important terms.
    """
    text = re.sub(r"[^A-Za-z0-9 ]+", " ", text)
    words = text.lower().split()

    stopwords = {
        "the","a","an","is","was","were","this","that","to","of","for","in","on",
        "at","and","it","by","as","with","from","will","be","new","about","has",
        "have","had","are","but","not","they","them","their","its","after","before",
        "says","claim","reported","news"
    }

    keywords = [w for w in words if w not in stopwords and len(w) > 3]

    clean = []
    for w in keywords:
        if w not in clean:
            clean.append(w)

    return " ".join(clean[:max_words])


# ------------------------------------------------
# CALL ZENSERP GOOGLE NEWS API
# ------------------------------------------------
def call_zenserp(query):
    """
    Calls Zenserp Google News API and returns a list of article results.
    """

    if not ZENSERP_API_KEY:
        raise ValueError("❌ Missing ZENSERP_API_KEY in .env")

    url = "https://app.zenserp.com/api/v2/search"
    params = {
        "apikey": ZENSERP_API_KEY,
        "q": query,
        "tbm": "nws",
        "num": 10
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        return data.get("news_results", [])

    except Exception as e:
        print("Zenserp API Error:", e)
        return []


# ------------------------------------------------
# RERANK ARTICLES BASED ON CLAIM SIMILARITY
# ------------------------------------------------
def rerank_articles(query, articles):
    """
    Ranks articles based on how similar they are to the given query.
    """
    query_words = set(query.lower().split())
    scored = []

    for art in articles:
        title = (art.get("title") or "").lower()
        desc = (art.get("snippet") or "").lower()

        title_score = sum(1 for w in query_words if w in title)
        desc_score = sum(1 for w in query_words if w in desc)
        total = title_score * 2 + desc_score  # title is more important

        scored.append((total, art))

    # Sort by best match
    scored.sort(key=lambda x: x[0], reverse=True)

    # Filter if there are any non-zero matches
    filtered = [a for score, a in scored if score > 0]
    return filtered if filtered else [a for _, a in scored]


# ------------------------------------------------
# MASTER FETCH FUNCTION (MAIN FUNCTION)
# ------------------------------------------------
def fetch_articles(raw_text, page_size=4):
    """
    Main function:
    1. Try full claim search
    2. If few results, fallback to keyword search
    3. Rerank using similarity
    4. Clean + return top results
    """

    # Step 1 — Try full claim search
    articles = call_zenserp(raw_text)

    # Step 2 — Fallback to keyword search
    if len(articles) < 2:
        keywords = extract_keywords(raw_text)
        if keywords:
            articles = call_zenserp(keywords)

    # Step 3 — Rerank results
    articles = rerank_articles(raw_text, articles)

    # Step 4 — Prepare clean output
    clean = []
    for a in articles[:page_size]:
        clean.append({
            "title": a.get("title"),
            "description": a.get("snippet") or "",
            "url": a.get("link"),
            "source": a.get("source"),
            "date": a.get("date")
        })

    return clean
