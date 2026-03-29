
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load('sentiment_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data['text']

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    return jsonify({"sentiment": prediction})

@app.route('/')
def home():
    return "API funcionando"
