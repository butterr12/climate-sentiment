from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

#svm ------
from pydantic import BaseModel
import joblib
import numpy as np

svm_model = joblib.load("svm_model/model.joblib")
svm_vectorizer = joblib.load("svm_model/vectorizer.joblib")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://climate-sentiment.vercel.app", 
        "http://localhost:5173",  
        "http://localhost:5174"
    ],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str
    model: str  # 'svm' or 'bert'

tokenizer = BertTokenizer.from_pretrained("./model")
model = BertForSequenceClassification.from_pretrained("./model", use_safetensors=True)
model.eval()

label_mapping = {
    0: "Risk",       
    1: "Neutral",     
    2: "Opportunity", 
}

@app.post("/predict")
async def predict(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    model_choice = input.model.lower()

    if model_choice == "svm":
        vector = svm_vectorizer.transform([input.text])
        pred = svm_model.predict(vector)[0]
        confidence = np.max(svm_model.decision_function(vector))
        label = label_mapping.get(prediction, f"LABEL_{pred}")
        return {
            "model": "svm",
            "prediction": label,
            "confidence": round(confidence, 2)
        }
    
    elif model_choice == "bert":
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            prediction = int(torch.argmax(probs, dim=-1).item())
            confidence = float(probs[0][prediction].item()) * 100

        label = label_mapping.get(prediction, f"LABEL_{prediction}")

        return {
            "model": "bert",
            "prediction": label,
            "confidence": round(confidence, 2)
        }
