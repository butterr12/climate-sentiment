from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import BertTokenizer, BertForSequenceClassification
import torch
import torch.nn.functional as F

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

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

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)
        prediction = int(torch.argmax(probs, dim=-1).item())
        confidence = float(probs[0][prediction].item()) * 100

    label = label_mapping.get(prediction, f"LABEL_{prediction}")

    return {
        "prediction": label,
        "confidence": round(confidence, 2)
    }
