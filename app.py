import streamlit as st
import pickle

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Title
st.title("🎫 IT Support Ticket Classifier")

st.write("Enter your issue below and get predicted category")

# Input box
user_input = st.text_area("Enter Ticket Text")

# Predict button
if st.button("Predict"):
    if user_input.strip() != "":
        vectorized_input = vectorizer.transform([user_input])
        prediction = model.predict(vectorized_input)

        st.success(f"Predicted Category: {prediction[0]}")
    else:
        st.warning("Please enter some text")
