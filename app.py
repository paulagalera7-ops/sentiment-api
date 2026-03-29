from flask import Flask, request, jsonify
import joblib
import os
from googletrans import Translator

app = Flask(__name__)

# Cargar modelo y vectorizador
base_path = os.path.dirname(__file__)
model = joblib.load(os.path.join(base_path, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(base_path, 'vectorizer.pkl'))

# Traductor
translator = Translator()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Obtener texto (Dialogflow o prueba manual)
    if 'queryResult' in data:
        text = data['queryResult']['queryText']
    elif 'text' in data:
        text = data['text']
    else:
        return jsonify({"error": "No text provided"}), 400

    # 🔥 TRADUCIR A INGLÉS (CLAVE)
    translated = translator.translate(text, dest='en').text

    # Debug (opcional pero útil)
    print("TEXTO ORIGINAL:", text)
    print("TRADUCIDO:", translated)

    # Vectorizar y predecir
    text_vec = vectorizer.transform([translated])
    prediction = model.predict(text_vec)[0]

    print("PREDICCIÓN:", prediction)

    # Respuestas según sentimiento
    if prediction.lower() == "negative":
        response = "Lamentamos lo ocurrido. Vamos a derivarte con un agente humano para ayudarte lo antes posible."
    
    elif prediction.lower() == "positive":
        response = "Nos alegra mucho tu experiencia. Podemos ofrecerte un descuento en tu próxima reserva."
    
    else:
        response = "Gracias por tu comentario. Si necesitas algo más, aquí estamos para ayudarte."

    return jsonify({
        "fulfillmentText": response
    })

@app.route('/')
def home():
    return "API funcionando"
