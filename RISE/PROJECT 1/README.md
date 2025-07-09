# Email Spam Detection

## Overview
This project implements a lightweight **Email Spam Detection** system using machine learning to classify emails as **spam** or **ham** (non-spam). The classifier achieves **98%+ accuracy** on the UCI SMS Spam Collection dataset and is designed for integration into web forms to enhance productivity and security by filtering spam emails.

### Problem Statement
Spam emails reduce productivity and pose security risks. This project aims to automatically classify emails as spam or ham using machine learning algorithms.

### Objective
Build a classifier using Naive Bayes with text preprocessing and TF-IDF vectorization, achieving 90%+ accuracy, and deployable in web applications.

## Features
- **Dataset**: Uses the UCI SMS Spam Collection dataset (`spam.csv`) with 5572 labeled emails (spam/ham).
- **Preprocessing**: 
  - Converts text to lowercase.
  - Removes special characters and digits.
  - Tokenizes text using NLTK's `punkt` tokenizer.
  - Removes stopwords to reduce noise.
- **Model**: Multinomial Naive Bayes classifier with TF-IDF vectorization (`TfidfVectorizer`).
- **Evaluation**: Reports accuracy, precision, recall, and a detailed classification report.
- **Output**: A trained model (`spam_classifier.pkl`) ready for web integration.
- **Example Predictions**: Classifies sample emails (e.g., "Win a free lottery..." as spam, "Will u meet ur dream partner soon? Is ur career off 2 a flyng start? 2 find out free, txt HORO followed by ur star sign, e. g. HORO ARIES" as ham).

## Requirements
- **Python**: 3.8 or higher
- **Libraries**:
  ```bash
  pip install pandas numpy scikit-learn nltk joblib
