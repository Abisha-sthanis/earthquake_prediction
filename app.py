from flask import Flask, render_template, request
import numpy as np
import os

app = Flask(__name__)

# Initialize model and scaler as None (for error handling)
model = None
scaler = None

try:
    from keras.models import load_model
    import joblib
    # Load model and scalers
    model = load_model('earthquake_lstm_model.keras')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Running in demo mode without ML model")

@app.route('/')
def welcome():
    return render_template('welcome.html')

# Add GET method for input page
@app.route('/input', methods=['GET'])
def input_form():
    return render_template('input.html')

@app.route('/input', methods=['POST'])
def input_page():
    name = request.form.get('name', 'User')
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data with error handling
        data = [
            float(request.form.get('longitude', 0)),
            float(request.form.get('latitude', 0)),
            float(request.form.get('depth', 0)),
            float(request.form.get('rms', 0)),
            float(request.form.get('type', 0)),
            int(request.form.get('date', 1)),
            int(request.form.get('month', 1)),
            int(request.form.get('year', 2024)),
            int(request.form.get('hour', 0)),
            int(request.form.get('minute', 0)),
            int(request.form.get('second', 0))
        ]

        # Check if model is loaded
        if model is None or scaler is None:
            # Demo prediction if model not loaded
            pred = np.random.uniform(2.0, 7.0)
        else:
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

        return render_template('output.html', 
                             magnitude=round(pred, 2), 
                             severity=severity, 
                             emoji=emoji)

    except Exception as e:
        return f"Error: {str(e)}"

# Add a simple test route
@app.route('/test')
def test():
    return "Flask is working!"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)