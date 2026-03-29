from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

import os

base_path = os.path.dirname(__file__)

model = joblib.load(os.path.join(base_path, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    if 'queryResult' in data:
        text = data['queryResult']['queryText']
    elif 'text' in data:
        text = data['text']
    else:
        return jsonify({"error": "No text provided"}), 400

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == "negative":
        response = "Lamento mucho tu experiencia. ¿Quieres ayuda?"
    elif prediction == "positive":
        response = "Qué bueno escuchar eso. ¿Te ayudo con algo más?"
    else:
        response = "Gracias por tu comentario."

    return jsonify({
        "fulfillmentText": response
    })

@app.route('/')
def home():
    return "API funcionando"
