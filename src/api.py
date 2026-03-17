from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

app = FastAPI()

model_path = "models/bert_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

class Claim(BaseModel):
    text: str


@app.get("/")
def home():
    return {"message": "Misinformation Detection API"}


@app.post("/predict")
def predict(claim: Claim):

    inputs = tokenizer(
        claim.text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

    label = "REAL" if prediction == 1 else "FAKE"

    return {
        "prediction": label,
        "confidence": round(confidence, 3)
    }
