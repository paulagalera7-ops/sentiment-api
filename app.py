@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    text = data['queryResult']['queryText']

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == "negative":
        response = "😔 Lamento mucho tu experiencia. Puedo ayudarte ahora mismo o derivarte a un agente humano. ¿Qué prefieres?"

    elif prediction == "positive":
        response = "😊 ¡Qué bueno escuchar eso! ¿Te gustaría ver más opciones o recomendaciones similares?"

    else:
        response = "🙂 Gracias por tu comentario. ¿Hay algo específico en lo que te pueda ayudar?"

    return jsonify({
        "fulfillmentText": response
    })
