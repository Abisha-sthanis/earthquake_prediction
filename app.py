# app.py
from flask import Flask, render_template, request
import numpy as np
import joblib
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load model and scalers
model = load_model('earthquake_lstm_model.keras')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    name = request.form['name']
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = [
            float(request.form['longitude']),
            float(request.form['latitude']),
            float(request.form['depth']),
            float(request.form['rms']),
            float(request.form['type']),
            int(request.form['date']),
            int(request.form['month']),
            int(request.form['year']),
            int(request.form['hour']),
            int(request.form['minute']),
            int(request.form['second'])
        ]

        # Scale and reshape for LSTM
        data_scaled = scaler.transform([data])
        data_scaled = data_scaled.reshape((1, 1, 11))

        pred = model.predict(data_scaled)[0][0]

        # Determine severity
        if pred < 3.0:
            severity = "Minor - Usually not felt"
            emoji = "🟢"
        elif pred < 4.0:
            severity = "Light - Often felt, but rarely causes damage"
            emoji = "🟡"
        elif pred < 5.0:
            severity = "Moderate - May cause damage to poorly constructed buildings"
            emoji = "🟠"
        elif pred < 6.0:
            severity = "Strong - Can cause damage in populated areas"
            emoji = "🔴"
        else:
            severity = "Major/Great - Serious damage over large areas"
            emoji = "🔴🔴"

        return render_template('output.html', magnitude=round(pred, 2), severity=severity, emoji=emoji)

    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)