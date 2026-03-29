@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    text = data['queryResult']['queryText']

    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]

    if prediction == "negative":
        response = "😔 Lamentamos mucho tu mala experiencia. Para solucionarlo puedo ayudarte ahora mismo mediante chat o derivarte a un agente humano. ¿Qué prefieres?"

    elif prediction == "positive":
        response = "😊 ¡Nos alegra escuchar eso! ¿Te gustaría ver más opciones similares en otras fecha o recomendaciones para ti en otra ciudades"

    else:
        response = "🙂 Gracias por tu comentario. ¿Hay algo más en lo que te pueda ayudar?"

    return jsonify({
        "fulfillmentText": response
    })
