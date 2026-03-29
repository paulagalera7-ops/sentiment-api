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

   from googletrans import Translator

translator = Translator()


translated = translator.translate(text, dest='en').text

print("TEXTO ORIGINAL:", text)
print("TRADUCIDO:", translated)


text_vec = vectorizer.transform([translated])
prediction = model.predict(text_vec)[0]

print("PREDICCIÓN:", prediction)

    if prediction == "negative":
        response = "Desde Booking lamentamos lo que nos comentas, para ofrecerte una rápida solucion te pondremos en contacto con un agente humano"
    elif prediction == "positive":
        response = "Nos alegra mucho que hayas tenida una muy buena experiencia, a continuacion te ofrecemos un descuento para alquiler de coche en tu próxima escapada "
    else:
        response = "Gracias por tu comentario, nos alegra contar con tu opinión siempre."

    return jsonify({
        "fulfillmentText": response
    })

@app.route('/')
def home():
    return "API funcionando"
