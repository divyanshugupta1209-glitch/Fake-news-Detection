from fastapi import FastAPI
from api.detection import router as detection_router

app = FastAPI(title="NEO-REFUTE Multimodal Fake News Detection")

app.include_router(detection_router, prefix="/predict", tags=["Prediction"])

@app.get("/")
def root():
    return {"message": "Welcome to NEO-REFUTE Multimodal Fake News Detection API"}
