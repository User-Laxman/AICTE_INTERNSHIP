import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import re

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Preprocessing function
def preprocess_text(text):
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

# Load and preprocess dataset
data = pd.read_csv('spam.csv')
data['Processed_Message'] = data['Message'].apply(preprocess_text)
data['Spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    data['Processed_Message'], data['Spam'], test_size=0.25, random_state=42
)

# Create pipeline with TF-IDF and Naive Bayes
clf = Pipeline([
    ('vectorizer', TfidfVectorizer(max_features=5000)),
    ('nb', MultinomialNB())
])

# Train the model
clf.fit(X_train, y_train)

# Evaluate the model
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# Test on example emails
emails = [
    'Sounds great! Are you home now?',
    'Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES'
]
predictions = clf.predict(emails)
print("\nExample Email Predictions:")
for email, pred in zip(emails, predictions):
    print(f"Email: {email}")
    print(f"Prediction: {'Spam' if pred == 1 else 'Ham'}\n")

# Save the model for web integration
import joblib
joblib.dump(clf, 'spam_classifier.pkl')