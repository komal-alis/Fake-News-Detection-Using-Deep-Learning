import streamlit as st
import pickle
import re
import nltk

from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================
# Download NLTK resources
# =========================
nltk.download('stopwords')
nltk.download('wordnet')

# =========================
# Load trained model
# =========================
model = load_model("model.h5", compile=False)

# Load vectorizer
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# =========================
# NLP setup
# =========================
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# =========================
# Preprocessing Function
# =========================
def preprocess(text):

    text = text.lower()

    # Remove special characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)

    # Tokenization
    words = text.split()

    # Stopword removal + Lemmatization
    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    return " ".join(words)

# =========================
# Streamlit Page Config
# =========================
st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="centered"
)

# =========================
# App Title
# =========================
st.title("📰 Fake News Detection System")

st.write(
    "Enter a news article below to check whether it is REAL or FAKE."
)

# =========================
# Sidebar
# =========================
st.sidebar.title("Model Information")

st.sidebar.write("Algorithm: Deep Learning")
st.sidebar.write("Feature Extraction: TF-IDF")
st.sidebar.write("Model: Neural Network")
st.sidebar.write("Epochs: 15")
st.sidebar.write("Dataset: Fake.csv + True.csv")

# =========================
# Text Input
# =========================
user_input = st.text_area(
    "Enter News Text",
    height=250
)

# =========================
# Prediction Button
# =========================
if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("⚠ Please enter news text.")

    else:

        # Preprocess text
        clean_text = preprocess(user_input)

        # Vectorize
        vector_input = vectorizer.transform([clean_text]).toarray()

        # Prediction
        prediction = model.predict(vector_input)[0][0]
        st.write("Prediction Score Raw =", prediction)
        # Show prediction value
        st.write(f"Prediction Score: {prediction:.4f}")

        # =========================
        # Correct Label Logic
        # =========================

        # 1 = Real
        # 0 = Fake

        if prediction >= 0.5:

            st.success("✅ REAL NEWS")

            confidence = prediction * 100

            st.write(f"Confidence: {confidence:.2f}%")

        else:

            st.error("🚨 FAKE NEWS")

            confidence = (1 - prediction) * 100

            st.write(f"Confidence: {confidence:.2f}%")

# =========================
# Example News
# =========================
st.subheader("Example News for Testing")

st.markdown("""
### Real News Example:
The Reserve Bank of India announced new digital payment guidelines to improve transaction security across the country.

### Fake News Example:
Scientists discovered a magical fruit that can cure every disease within 24 hours without medical treatment.
""")