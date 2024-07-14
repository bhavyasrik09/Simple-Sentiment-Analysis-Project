import nltk
from nltk.corpus import movie_reviews
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Download necessary NLTK data
nltk.download('movie_reviews')

# Load the dataset
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

random.shuffle(documents)

# Prepare the text and labels
texts = [' '.join(words) for words, category in documents]
labels = [category for words, category in documents]

# Convert labels to numerical values
y = np.array([1 if label == 'pos' else 0 for label in labels])

# Feature extraction using TF-IDF
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(texts).toarray()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(classification_report(y_test, y_pred, target_names=['neg', 'pos']))

# Save the model and vectorizer
joblib.dump(model, 'sentiment_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')

# Function to predict sentiment of new text
def predict_sentiment(text):
    features = vectorizer.transform([text]).toarray()
    prediction = model.predict(features)
    return 'pos' if prediction == 1 else 'neg'

# Test the function
text = "This movie was fantastic!"
print(predict_sentiment(text))
