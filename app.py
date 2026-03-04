import pickle
import string
import nltk
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

@st.cache_resource
def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("stopwords", quiet=True)

setup_nltk()


@st.cache_resource
def load_artifacts():
    tfidf = pickle.load(open("vectorizer.pkl", "rb"))
    model = pickle.load(open("model.pkl", "rb"))
    return tfidf, model

tfidf, model = load_artifacts()

ps = PorterStemmer()
STOP = set(stopwords.words("english"))

def transform_text(text):
    text = text.lower()
    words = nltk.word_tokenize(text)
    words = [i for i in words if i.isalnum()]
    words = [i for i in words if i not in STOP and i not in string.punctuation]
    words = [ps.stem(i) for i in words]
    return " ".join(words)

st.title("SMS Spam Detector")

message = st.text_area("Enter your message")

if st.button("Predict"):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        transformed = transform_text(message)
        vector = tfidf.transform([transformed])
        prediction = model.predict(vector)[0]

        if prediction == 1:
            st.error(" Spam Message")
        else:
            st.success(" Ham (Not Spam)")