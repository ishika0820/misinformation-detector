import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Load trained model
model_path = "models/bert_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

model.eval()

print("BERT Misinformation Detector")
print("-----------------------------")

while True:
    text = input("\nEnter a claim (or type 'quit'): ")

    if text.lower() == "quit":
        break

    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    prediction = torch.argmax(probs).item()
    confidence = probs[0][prediction].item()

    if prediction == 1:
        label = "REAL"
    else:
        label = "FAKE"

    print(f"Prediction: {label}")
    print(f"Confidence: {confidence:.2f}")
