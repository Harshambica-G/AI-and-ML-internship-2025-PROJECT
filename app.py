import streamlit as st
import pickle
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('stopwords')

# Load model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf.pkl", "rb"))

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    text = re.sub('[^a-zA-Z]', ' ', text)
    tokens = text.split()
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

st.title("📰 Fake News Classifier")
st.markdown(
    """
    <style>
    /* App background: subtle gradient navy to slate gray */
    .stApp {
        background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d);
        min-height: 100vh;
        color: #f0f0f0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    /* Title and headers: crisp white with subtle shadow */
    h1, h2, h3, h4, h5, h6 {
        color: #e0e0e0;
        text-shadow: 1px 1px 2px #000000aa;
    }

    /* Buttons: navy background with gold accent on hover */
    div.stButton > button {
        background-color: #1a2a6c;
        color: #fdbb2d;
        border-radius: 8px;
        border: 2px solid #fdbb2d;
        padding: 10px 25px;
        font-weight: 600;
        transition: background-color 0.3s ease, color 0.3s ease;
        cursor: pointer;
        box-shadow: 0 4px 6px rgba(26, 42, 108, 0.5);
    }
    div.stButton > button:hover {
        background-color: #fdbb2d;
        color: #1a2a6c;
        box-shadow: 0 6px 12px rgba(253, 187, 45, 0.7);
    }

    /* Text area: dark slate background with light text */
    textarea {
        background-color: #2f3e67;
        color: #e0e0e0;
        font-weight: 500;
        border-radius: 8px;
        padding: 10px;
        border: none;
        box-shadow: inset 0 0 5px #1a2a6c;
    }

    /* Placeholder text color */
    textarea::placeholder {
        color: #a0a0a0;
    }

    /* Margin below the title */
    .css-18e3th9 {
        margin-bottom: 30px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)





user_input = st.text_area("Enter News Article Text")

if st.button("Classify"):
    cleaned = preprocess(user_input)
    vect_text = vectorizer.transform([cleaned])
    prediction = model.predict(vect_text)[0]
    result = "✅ Real News" if prediction == 1 else "❌ Fake News"
    st.subheader(result)
