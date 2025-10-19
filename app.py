from flask import Flask, render_template, request
import numpy as np
import os
import traceback

app = Flask(__name__)

print("=== DEBUG: Starting app ===")
print(f"Current directory: {os.getcwd()}")
print(f"Template folder: {app.template_folder}")
print(f"Templates exist: {os.path.exists(app.template_folder)}")
if os.path.exists(app.template_folder):
    print(f"Files in templates: {os.listdir(app.template_folder)}")
print(f"Model exists: {os.path.exists('earthquake_lstm_model.keras')}")
print(f"Scaler exists: {os.path.exists('scaler.pkl')}")

# Initialize model and scaler as None
model = None
scaler = None

# Try to load model and scaler
try:
    from tensorflow.keras.models import load_model
    import joblib
    print("=== DEBUG: TensorFlow and joblib imported successfully ===")

    if os.path.exists('earthquake_lstm_model.keras') and os.path.exists('scaler.pkl'):
        model = load_model('earthquake_lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
    else:
        print("⚠️ Warning: Model files not found. Prediction will not work.")

except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    traceback.print_exc()

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    name = request.form.get('name', 'User')
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or scaler is None:
            raise Exception("Model files not loaded. Please check deployment logs.")

        # Collect input data
        name = request.form.get('name', 'User')
        longitude = float(request.form['longitude'])
        latitude = float(request.form['latitude'])
        depth = float(request.form['depth'])
        rms = float(request.form['rms'])
        type_val = int(request.form['type'])
        date = int(request.form['date'])
        month = int(request.form['month'])
        year = int(request.form['year'])
        hour = int(request.form['hour'])
        minute = int(request.form['minute'])
        second = int(request.form['second'])

        # Prepare input
        user_input = np.array([[longitude, latitude, depth, rms, type_val,
                                date, month, year, hour, minute, second]])
        user_input_scaled = scaler.transform(user_input)
        user_input_reshaped = user_input_scaled.reshape(1, 1, 11)

        # Predict
        prediction = model.predict(user_input_reshaped, verbose=0)
        predicted_magnitude = float(prediction[0][0])

        # Categorize result
        if predicted_magnitude < 3.0:
            severity, emoji, severity_class, risk_description = (
                "Minor - Usually not felt", "🟢", "severity-low",
                "NOTICE: Minor seismic activity predicted."
            )
        elif predicted_magnitude < 4.0:
            severity, emoji, severity_class, risk_description = (
                "Light - Often felt, but rarely causes damage", "🟡", "severity-low",
                "CAUTION: Light earthquake predicted."
            )
        elif predicted_magnitude < 5.0:
            severity, emoji, severity_class, risk_description = (
                "Moderate - May cause damage to poorly built buildings", "🟠", "severity-medium",
                "ALERT: Moderate earthquake predicted."
            )
        elif predicted_magnitude < 6.0:
            severity, emoji, severity_class, risk_description = (
                "Strong - Can cause damage in populated areas", "🔴", "severity-high",
                "WARNING: Strong earthquake predicted."
            )
        else:
            severity, emoji, severity_class, risk_description = (
                "Major/Great - Serious damage over large areas", "🔴", "severity-high",
                "CRITICAL: Major earthquake predicted."
            )

        magnitude_formatted = f"{predicted_magnitude:.2f}"

        return render_template(
            'output.html',
            name=name,
            magnitude=magnitude_formatted,
            emoji=emoji,
            severity=severity,
            severity_class=severity_class,
            risk_description=risk_description,
            longitude=f"{longitude:.4f}",
            latitude=f"{latitude:.4f}",
            depth=f"{depth:.2f}",
            rms=f"{rms:.2f}",
            type=type_val,
            date=f"{date:02d}",
            month=f"{month:02d}",
            year=year,
            hour=f"{hour:02d}",
            minute=f"{minute:02d}",
            second=f"{second:02d}"
        )

    except Exception as e:
        print("❌ Prediction Error:", e)
        traceback.print_exc()
        return f"Error: {str(e)}", 500


if __name__ == '__main__':
    # Get port from environment variable (for Render deployment)
    port = int(os.environ.get('PORT', 5000))
    
    print("\n" + "=" * 60)
    print("🌍 EARTHQUAKE PREDICTION SYSTEM")
    print("=" * 60)
    print("🚀 Starting Flask server...")
    print(f"📍 Port: {port}")
    print("=" * 60 + "\n")
    
    # For production (Render), don't use debug mode
    app.run(debug=False, host="0.0.0.0", port=port)