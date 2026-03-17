import streamlit as st
import requests

st.title("AI Misinformation Detector")

st.write("Enter a political claim to check if it may contain misinformation.")

claim = st.text_input("Enter a claim")

if st.button("Analyze"):

    response = requests.post(
        "http://127.0.0.1:8000/predict",
        json={"text": claim}
    )

    result = response.json()

    st.subheader("Result")

    st.write("Prediction:", result["prediction"])
    st.write("Confidence:", result["confidence"])
