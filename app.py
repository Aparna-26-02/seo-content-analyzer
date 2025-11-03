import streamlit as st
import pandas as pd
import joblib
import requests
from bs4 import BeautifulSoup
import re
import textstat

# --- 1. Load the Saved Model and Vectorizer ---
try:
    model = joblib.load('content_quality_model.joblib')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.joblib')
except FileNotFoundError:
    st.error("Model or vectorizer not found. Please run the notebook to create these files.")
    st.stop()

# --- 2. Define the Functions for Real-Time Analysis ---

def fetch_html(url):
    """Fetches HTML content from a URL."""
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        return response.text
    except requests.RequestException as e:
        st.error(f"Error fetching URL: {e}")
        return None

def parse_html(html_content):
    """Parses HTML to extract clean text."""
    if not html_content:
        return ""
    soup = BeautifulSoup(html_content, 'lxml')
    for script_or_style in soup(["script", "style"]):
        script_or_style.decompose()
    return soup.get_text(separator=' ', strip=True)

def calculate_features(text):
    """Calculates all the necessary features from the text."""
    # Basic features
    word_count = len(text.split())
    char_count = len(text)
    avg_word_length = 0
    if word_count > 0:
        avg_word_length = char_count / word_count
    
    # Readability score
    readability_score = 0
    try:
        if word_count > 100:
            readability_score = textstat.flesch_reading_ease(text)
    except:
        pass # Ignore errors for very short texts

    # Create a DataFrame for the model
    features = pd.DataFrame([[word_count, char_count, avg_word_length, readability_score]],
                            columns=['word_count', 'char_count', 'avg_word_length', 'readability_score'])
    return features

# --- 3. Build the Streamlit User Interface ---

st.title('🤖 SEO Content Quality Detector')
st.write("Enter a URL to analyze its content quality using our trained machine learning model.")

# URL input box
url_input = st.text_input('Enter the URL of an article:', '')

if st.button('Analyze Content'):
    if url_input:
        with st.spinner('Fetching and analyzing content... This might take a moment.'):
            # Step A: Fetch and Parse
            html = fetch_html(url_input)
            if html:
                cleaned_text = parse_html(html)

                # Step B: Engineer Features
                features_df = calculate_features(cleaned_text)
                
                # Step C: Make a Prediction
                prediction = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)

                # Step D: Display Results
                st.subheader("Analysis Results")
                if prediction[0] == 1:
                    st.success("High Quality Content Predicted!")
                else:
                    st.warning("Low Quality Content Predicted.")

                st.write(f"Confidence Score:")
                st.write(f"- **High Quality:** {prediction_proba[0][1]:.2%}")
                st.write(f"- **Low Quality:** {prediction_proba[0][0]:.2%}")
                
                st.subheader("Content Features")
                st.dataframe(features_df)

    else:
        st.warning("Please enter a URL to analyze.")