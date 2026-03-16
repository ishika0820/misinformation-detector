import pickle

# load model
model = pickle.load(open("models/model.pkl", "rb"))
vectorizer = pickle.load(open("models/vectorizer.pkl", "rb"))

print("Misinformation Detector")
print("------------------------")

while True:
    text = input("\nEnter a claim (or type 'quit'): ")

    if text.lower() == "quit":
        break

    text_vec = vectorizer.transform([text])

    prediction = model.predict(text_vec)[0]

    if prediction == 1:
        print("Prediction: REAL")
    else:
        print("Prediction: FAKE")
