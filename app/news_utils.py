# backend/news_utils.py
import os, requests
from dotenv import load_dotenv
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def fetch_related_articles(query: str, page_size: int = 3):
    """Return list of article dicts (title, source, url, description)."""
    if not NEWS_API_KEY:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": query,
        "language": "en",
        "pageSize": page_size,
        "sortBy": "relevancy",
        "apiKey": NEWS_API_KEY
    }
    try:
        r = requests.get(url, params=params, timeout=8)
        r.raise_for_status()
        data = r.json()
        return data.get("articles", [])
    except Exception:
        return []
