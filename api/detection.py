from fastapi import APIRouter
from pydantic import BaseModel
from app.model import classify
from app.newsapi_client import search_news

router = APIRouter()

class NewsItem(BaseModel):
    title: str
    content: str
    image_url: str = None

@router.post("/")
def predict_news(item: NewsItem):
    text = item.title + " " + item.content
    prediction = classify(text, item.image_url)
    external_refs = search_news(item.title)
    return {"prediction": prediction, "external_check": external_refs}
