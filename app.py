from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import os
from download_model import *
app = FastAPI()
if not os.path.exists(
    "sentiment_model/model.safetensors"
):
    import download_model


classifier = pipeline(
    "sentiment-analysis",
    model="sentiment_model",
    tokenizer="sentiment_model"
)
class Review(BaseModel):
    text: str

# Home endpoint
@app.get("/")
def home():
    return {"message": "Sentiment Analysis API is Running!"}

# Prediction endpoint
@app.post("/predict")
def predict(review: Review):

    result = classifier(review.text)[0]

    sentiment = (
        "Positive"
        if result["label"] == "LABEL_1"
        else "Negative"
    )

    return {
        "review": review.text,
        "sentiment": sentiment,
        "confidence": round(result["score"] * 100, 2)
    }