import os
import pickle
import string

import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


@st.cache_resource
def setup_nltk():
    # Store NLTK data inside the project folder so cloud can access it
    nltk_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
    os.makedirs(nltk_dir, exist_ok=True)
    if nltk_dir not in nltk.data.path:
        nltk.data.path.append(nltk_dir)

    # Needed resources (punkt_tab is required in newer NLTK for sentence tokenization)
    nltk.download("punkt", download_dir=nltk_dir, quiet=True)
    nltk.download("punkt_tab", download_dir=nltk_dir, quiet=True)
    nltk.download("stopwords", download_dir=nltk_dir, quiet=True)

setup_nltk()


@st.cache_resource
def load_artifacts():
    with open("vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return tfidf, model

tfidf, model = load_artifacts()



ps = PorterStemmer()
STOP = set(stopwords.words("english"))

def transform_text(text: str) -> str:
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [w for w in words if w.isalnum()]
    words = [w for w in words if w not in STOP and w not in string.punctuation]
    words = [ps.stem(w) for w in words]
    return " ".join(words)


st.title("Email/SMS Spam Detector")

message = st.text_area("Enter your message")

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        transformed = transform_text(message)
        vector = tfidf.transform([transformed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error(" SPAM")
        else:
            st.success(" HAM (Not Spam)")