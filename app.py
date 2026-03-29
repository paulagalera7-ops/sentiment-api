@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # 🔥 Manejo correcto para Dialogflow + pruebas manuales
    if 'queryResult' in data:
        text = data['queryResult']['queryText']
    elif 'text' in data:
        text = data['text']
    else:
        return jsonify({"error": "No text provided"}), 400

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == "negative":
        response = "😔 Lamento mucho tu experiencia. ¿Como puedo ayudarte?"
    elif prediction == "positive":
        response = "😊 ¡Qué bueno escuchar eso!"
    else:
        response = "🙂 Gracias por tu comentario positivo. Nos alegra que sigas contando con nuestros servicios en el futuro "

    return jsonify({
        "fulfillmentText": response
    })
