import streamlit as st
from transformers import pipeline

# Load SLM
@st.cache_resource
def load_model():
    return pipeline("text-generation", model="distilbert-base-uncased")

model = load_model()

# UI
st.title("SLM Text Generator")
user_input = st.text_input("Enter a prompt:")
if user_input:
    result = model(user_input, max_length=50, num_return_sequences=1)
    st.write("Generated Text:", result[0]['generated_text'])
