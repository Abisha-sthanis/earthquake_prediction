from flask import Flask, render_template, request
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Global variables
model = None
scaler = None

# Load model with error handling
def load_models():
    global model, scaler
    try:
        model_path = 'earthquake_lstm_model.keras'
        scaler_path = 'scaler.pkl'
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            print("✅ Model loaded successfully!")
        else:
            print(f"❌ Model file not found: {model_path}")
            
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            print("✅ Scaler loaded successfully!")
        else:
            print(f"❌ Scaler file not found: {scaler_path}")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")

# Load models on startup
load_models()

@app.route('/', methods=['GET', 'HEAD'])
def welcome():
    """Welcome page - handles both GET and HEAD requests"""
    if request.method == 'HEAD':
        return '', 200
    print("=== DEBUG: Welcome page accessed ===")
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    """Receive user name and show input form"""
    name = request.form.get('name', 'User')
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle earthquake prediction"""
    try:
        # Check if models are loaded
        if model is None or scaler is None:
            return render_template('output.html', 
                                 error="Model files not loaded. Please ensure earthquake_lstm_model.keras and scaler.pkl are present.")
        
        # Get and validate form data
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

        # Scale and reshape for LSTM
        data_scaled = scaler.transform([data])
        data_scaled = data_scaled.reshape((1, 1, 11))

        # Make prediction
        pred = model.predict(data_scaled, verbose=0)[0][0]

        # Determine severity
        if pred < 3.0:
            severity = "Minor - Usually not felt"
            emoji = "🟢"
            color = "success"
        elif pred < 4.0:
            severity = "Light - Often felt, but rarely causes damage"
            emoji = "🟡"
            color = "warning"
        elif pred < 5.0:
            severity = "Moderate - May cause damage to poorly constructed buildings"
            emoji = "🟠"
            color = "orange"
        elif pred < 6.0:
            severity = "Strong - Can cause damage in populated areas"
            emoji = "🔴"
            color = "danger"
        else:
            severity = "Major/Great - Serious damage over large areas"
            emoji = "🔴🔴"
            color = "critical"

        return render_template('output.html', 
                             magnitude=round(pred, 2), 
                             severity=severity, 
                             emoji=emoji,
                             color=color,
                             longitude=data[0],
                             latitude=data[1],
                             depth=data[2],
                             rms=data[3],
                             type_val=data[4],
                             date=data[5],
                             month=data[6],
                             year=data[7],
                             hour=data[8],
                             minute=data[9],
                             second=data[10])

    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return render_template('output.html', error=f"Prediction failed: {str(e)}")

@app.route('/health')
def health():
    """Health check endpoint"""
    return {
        'status': 'healthy',
        'model_loaded': model is not None,
        'scaler_loaded': scaler is not None
    }, 200

if __name__ == '__main__':
    # Get port from environment variable (Render/Docker) or use 10000
    port = int(os.environ.get('PORT', 10000))
    # Run on 0.0.0.0 to be accessible from outside container
    app.run(host='0.0.0.0', port=port, debug=False)