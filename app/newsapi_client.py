import os
import requests

NEWSAPI_KEY = os.getenv("NEWSAPI_KEY")

def search_news(query: str, max_results=3):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={NEWSAPI_KEY}"
    response = requests.get(url)
    results = []
    if response.status_code == 200:
        data = response.json()
        for article in data.get("articles", [])[:max_results]:
            results.append({"title": article["title"], "source": article["source"]["name"]})
    return results
