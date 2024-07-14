from flask import Flask, request, render_template
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

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        text = request.form['text']
        sentiment = predict_sentiment(text)
        return render_template('index.html', prediction_text=f"The sentiment of the text is: {sentiment}")

if __name__ == "__main__":
    app.run(debug=True)
