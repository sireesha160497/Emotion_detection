# WebInterface.py
from flask import Flask, request, jsonify
from EmotionDetectionModel import EmotionDetectionModel

app = Flask(__name__)
model = EmotionDetectionModel('train.csv')
model.train_model()


@app.route('/predict', methods=['POST'])
def predict_emotion():
    if request.method == 'POST':
        data = request.json
        if data and 'text' in data:
            text = data['text']
            predicted_emotion = model.predict_emotion(text)
            return jsonify({'predicted_emotion': predicted_emotion}), 200
        else:
            return jsonify({'error': 'Invalid request. Please provide text parameter.'}), 400
    else:
        return jsonify({'error': 'Invalid request method. Only POST requests are supported.'}), 405


if __name__ == '__main__':
    app.run(debug=False)
