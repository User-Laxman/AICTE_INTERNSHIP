import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,precision_score, recall_score, classification_report
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re
import joblib

nltk.download("punkt_tab")
nltk.download("stopwords")

# Preprocessing function
def preprocess_text(text):
    try:
        # Convert to lowercase
        text = text.lower()
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
        # Join tokens back to string
        return ' '.join(tokens)
    except Exception as e:
        print(f"Error in preprocessing: {e}")
        return text

#map indices to emotions 
emotions = {
        0: 'sadness',
        1: 'joy',
        2: 'love',
        3: 'anger',
        4: 'fear',
        5: 'surprise'
}

# Load dataset (placeholder for Emotion Dataset)
def load_dataset():
    data = pd.read_csv("emotions.csv")
    data.columns
    data["processed_text"] = data["text"].apply(preprocess_text)
    return data

# Main function for emotion detection
def train_emotion_classifier():
    # Load and preprocess data
    df = load_dataset()
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['label'], test_size=0.2, random_state=42
    )
    
    # Create pipeline with TF-IDF and Logistic Regression
    clf = Pipeline([
        ('vectorizer', TfidfVectorizer(max_features=5000)),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    
    # Train the model
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    
    print("Model Evaluation Metrics:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return clf

# Function to predict emotion on new text
def predict_emotion(text, model):
    processed_text = preprocess_text(text)
    prediction = model.predict([processed_text])[0]
    return prediction

# Train the model
model = train_emotion_classifier()
    
# Save the model for integration
joblib.dump(model, 'emotion_classifier.pkl')
    
# Example predictions
test_texts = [
        "i was beaten by a dog.",
        "Feeling so down after the news."
]
print("\nExample Text Predictions:")
for text in test_texts:
    emotion = predict_emotion(text, model)
    print(f"Text: {text}")
    print(f"Predicted Emotion: {emotions[emotion]}\n")