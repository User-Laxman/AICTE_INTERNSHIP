# Emotion Detection from Text

## Overview
Classifies emotions (e.g., happy, sad, angry) from text messages using Logistic Regression and TF-IDF vectorization. Designed for integration into chatbots or feedback analyzers for education and mental health applications.

## Features
- Preprocesses text (lowercase, remove special characters, tokenize, remove stopwords).
- Uses TF-IDF vectorization and Logistic Regression for multi-class emotion classification.
- Evaluates with accuracy, precision, recall, and detailed classification report.
- Saves model (`emotion_classifier.pkl`) for web integration.

## Requirements
- Python 3.8+
- Libraries: `pandas`, `numpy`, `scikit-learn`, `nltk`, `joblib`
- Dataset: Emotion-labeled text (e.g., [Kaggle Emotion Dataset](https://www.kaggle.com/datasets/bhavikjikadara/emotions-dataset))
- NLTK resources: `punkt`, `punkt_tab`, `stopwords`

## Usage
Run: `python emotion_detection.py`
- Trains model on the dataset.
- Outputs evaluation metrics and example predictions.
- Saves `emotion_classifier.pkl` for chatbot integration.
