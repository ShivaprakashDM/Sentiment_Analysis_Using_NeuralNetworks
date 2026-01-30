# Sentiment Analysis & Clustering Web Application

This project is a Flask-based web application that performs **Sentiment Analysis** and **Text Clustering** on user inputs (tweets, comments, reviews). It leverages Machine Learning models to predict sentiment polarity (Positive, Neutral, Negative) and groups similar text using Unsupervised Learning (K-Means).

## 📌 Project Overview

The core objective of this application is to interpret and categorize textual data. It uses a pre-trained Support Vector Machine (or similar classifier stored in `clf.pkl`) for sentiment prediction and a K-Means algorithm (trained on TF-IDF vectors) to cluster text into semantically similar groups.

### Key Features
*   **Real-time Sentiment Detection**: Instantly classifies text as **Positive**, **Negative**, or **Neutral**.
*   **Unsupervised Clustering**: Assigns the input text to a specific cluster (0-3) based on its content similarity.
*   **Text Preprocessing Pipeline**: Automatically cleans data by removing HTML tags, handling emojis, stemming words, and removing stopwords.
*   **Web Interface**: User-friendly interface built with Flask and HTML/CSS.

## 🛠️ Tech Stack

*   **Language**: Python 3.x
*   **Framework**: Flask
*   **Machine Learning**:
    *   Scikit-learn (KMeans, TfidfVectorizer)
    *   Pickle (Model serialization)
*   **NLP Libraries**:
    *   NLTK (Stopwords, PorterStemmer)
    *   Re (Regular Expressions for text cleaning)
*   **Data Handling**: Pandas

## 📂 Repository Structure

```
Sentiment-Analysis/
├── app.py                  # Main application entry point (Flask)
├── clf.pkl                 # Pre-trained Classification Model
├── tfidf.pkl               # Pre-trained TF-IDF Vectorizer Model
├── Tweets.csv              # Dataset used for training/clustering
├── Sentiment_Analysis.ipynb # Jupyter Notebook for model training & analysis
├── templates/
│   └── index.html          # Frontend HTML template
└── README.md               # Project Documentation
```

## 🚀 Getting Started

### Prerequisites
*   Python 3.7+
*   pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ShivaprakashDM/Sentiment_Analysis_Using_NeuralNetworks.git
    cd Sentiment_Analysis_Using_NeuralNetworks
    ```

2.  **Install required packages:**
    ```bash
    pip install flask pandas nltk scikit-learn
    ```

3.  **Download NLTK data (if not already present):**
    Open a python shell and run:
    ```python
    import nltk
    nltk.download('stopwords')
    ```

### Configuration
*   **Note**: The file `app.py` may contain absolute paths (e.g., `C:\Users\DELL\...`). Please update the `data_path` variable in `app.py` to point to the correct location of `Tweets.csv` on your machine, or use a relative path like `pd.read_csv('Tweets.csv', ...)`.

### Running the App

1.  Start the Flask server:
    ```bash
    python app.py
    ```
2.  Open your web browser and navigate to:
    `http://127.0.0.1:5000/`

## 🧠 How It Works

1.  **Preprocessing**: The input text undergoes cleaning (lowercase, regex removal of special chars, emoji handling) and stemming using NLTK's PorterStemmer.
2.  **Vectorization**: The processed text is converted into a numerical vector using the pre-loaded `tfidf.pkl` (TF-IDF Vectorizer).
3.  **Prediction**: 
    *   The `clf.pkl` model predicts the sentiment probability.
    *   The `KMeans` model assigns a cluster ID.
4.  **Result**: The result is rendered on the web page.
