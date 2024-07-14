import joblib

# Load the model and vectorizer
model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

def predict_sentiment(text):
    # Transform the text to feature vectors
    features = vectorizer.transform([text]).toarray()
    # Predict the sentiment
    prediction = model.predict(features)
    return 'Positive' if prediction == 1 else 'Negative'

if __name__ == "__main__":
    # Take text input from the user
    text = input("Enter the text to analyze sentiment: ")
    # Predict the sentiment
    sentiment = predict_sentiment(text)
    # Print the sentiment
    print(f"The sentiment of the text is: {sentiment}")
