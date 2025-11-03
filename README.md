# SEO Content Quality & Duplicate Detector

## 📝 Project Overview

This project is a machine learning pipeline designed to analyze web content for SEO quality and detect near-duplicate articles. It processes raw HTML, engineers NLP features, and uses a classification model to predict content quality. This was submitted as a data science assignment.

---

## 🚀 Features

- **HTML Parsing**: Extracts clean, readable text from raw HTML content.
- **NLP Preprocessing**: Cleans and tokenizes text for analysis.
- **Feature Engineering**: Calculates features like word count, character count, and readability scores (Flesch Reading Ease).
- **Duplicate Detection**: Uses TF-IDF and Cosine Similarity to identify articles with high textual similarity.
- **Quality Prediction**: Employs a trained Logistic Regression model to classify content as "High Quality" or "Low Quality" with **94% accuracy**.
- **(Bonus) Interactive Web App**: A Streamlit application for real-time analysis of any given URL.

---

## 🛠️ Tech Stack

- Python 3.9+
- Pandas
- Scikit-learn
- NLTK & Textstat
- BeautifulSoup4
- Streamlit (for the bonus web app)

---

## 🏁 How to Run

### 1. Set up the Environment

First, clone the repository and set up the virtual environment:

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
python -m venv venv
source venv/bin/activate  # On macOS/Linux
# .\venv\Scripts\activate    # On Windows