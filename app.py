from flask import Flask, render_template, request
import numpy as np
import os
import sys

app = Flask(__name__)

# Initialize model, scaler as None
model = None
scaler = None

print("=" * 60)
print("🌍 EARTHQUAKE PREDICTION SYSTEM - STARTING")
print("=" * 60)
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Files in directory: {os.listdir('.')}")
print("=" * 60)

# Try to load model files
try:
    print("📦 Attempting to load TensorFlow and Keras...")
    from tensorflow.keras.models import load_model
    print("✅ TensorFlow imported successfully")
    
    print("📦 Attempting to load joblib...")
    import joblib
    print("✅ Joblib imported successfully")
    
    # Check if model files exist
    model_path = 'earthquake_lstm_model.keras'
    scaler_path = 'scaler.pkl'
    
    print(f"\n🔍 Checking for model file: {model_path}")
    if os.path.exists(model_path):
        print(f"✅ Found {model_path} (Size: {os.path.getsize(model_path)} bytes)")
        model = load_model(model_path)
        print("✅ Model loaded successfully!")
    else:
        print(f"❌ {model_path} not found!")
        print(f"Files in current directory: {[f for f in os.listdir('.') if f.endswith('.keras')]}")
    
    print(f"\n🔍 Checking for scaler file: {scaler_path}")
    if os.path.exists(scaler_path):
        print(f"✅ Found {scaler_path} (Size: {os.path.getsize(scaler_path)} bytes)")
        scaler = joblib.load(scaler_path)
        print("✅ Scaler loaded successfully!")
    else:
        print(f"❌ {scaler_path} not found!")
        print(f"Files in current directory: {[f for f in os.listdir('.') if f.endswith('.pkl')]}")
    
    if model and scaler:
        print("\n" + "=" * 60)
        print("✅ ALL MODEL FILES LOADED SUCCESSFULLY!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("⚠️  WARNING: Some model files are missing!")
        print("=" * 60)
        
except Exception as e:
    print(f"\n❌ ERROR loading model files: {str(e)}")
    print(f"Error type: {type(e).__name__}")
    import traceback
    traceback.print_exc()
    print("=" * 60)

@app.route('/')
def welcome():
    """Welcome page - Get user's name"""
    print("📍 Route: / (welcome page)")
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    """Input page - Show form with user's name"""
    name = request.form.get('name', 'User')
    print(f"📍 Route: /input (name={name})")
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    """Process prediction and show results"""
    print("📍 Route: /predict")
    try:
        # Check if model is loaded
        if model is None or scaler is None:
            error_msg = "Model files not loaded. "
            if model is None:
                error_msg += "earthquake_lstm_model.keras is missing. "
            if scaler is None:
                error_msg += "scaler.pkl is missing. "
            error_msg += "Please ensure both files are in the deployment directory."
            raise Exception(error_msg)
        
        # Get user name
        name = request.form.get('name', 'User')
        
        # Get all input values from form
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
        
        print(f"🔢 Input received: lon={longitude}, lat={latitude}, depth={depth}")
        
        # Create input array (must match training feature order)
        user_input = np.array([[longitude, latitude, depth, rms, type_val, 
                               date, month, year, hour, minute, second]])
        
        # Scale the input using the saved scaler
        user_input_scaled = scaler.transform(user_input)
        
        # Reshape for LSTM [samples, timesteps, features]
        user_input_reshaped = user_input_scaled.reshape(1, 1, 11)
        
        # Make prediction
        print("🤖 Making prediction...")
        prediction = model.predict(user_input_reshaped, verbose=0)
        predicted_magnitude = float(prediction[0][0])
        print(f"✅ Prediction: {predicted_magnitude:.2f}")
        
        # Classify earthquake severity
        if predicted_magnitude < 3.0:
            severity = "Minor - Usually not felt"
            emoji = "🟢"
            severity_class = "severity-low"
            risk_description = "NOTICE: Minor seismic activity predicted. Little to no damage expected. Normal monitoring procedures sufficient."
        elif predicted_magnitude < 4.0:
            severity = "Light - Often felt, but rarely causes damage"
            emoji = "🟡"
            severity_class = "severity-low"
            risk_description = "CAUTION: Light earthquake predicted. Often felt but rarely causes damage. Most buildings will withstand the shaking."
        elif predicted_magnitude < 5.0:
            severity = "Moderate - May cause damage to poorly constructed buildings"
            emoji = "🟠"
            severity_class = "severity-medium"
            risk_description = "ALERT: Moderate earthquake predicted. Notable shaking expected with potential for minor structural damage. Secure loose objects and stay alert."
        elif predicted_magnitude < 6.0:
            severity = "Strong - Can cause damage in populated areas"
            emoji = "🔴"
            severity_class = "severity-high"
            risk_description = "WARNING: Strong earthquake predicted. Significant damage to buildings expected. Implement safety measures and prepare for aftershocks."
        else:
            severity = "Major/Great - Serious damage over large areas"
            emoji = "🔴"
            severity_class = "severity-high"
            risk_description = "CRITICAL: Major earthquake predicted. Immediate evacuation and emergency protocols should be activated. Expect severe structural damage and potential casualties."
        
        # Format the magnitude to 2 decimal places
        magnitude_formatted = f"{predicted_magnitude:.2f}"
        
        # Render output template with all data
        return render_template('output.html',
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
                             second=f"{second:02d}")
    
    except Exception as e:
        print(f"❌ Error in prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return f"""
        <html>
            <head>
                <title>Error</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        min-height: 100vh;
                        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                        margin: 0;
                    }}
                    .error-box {{
                        background: white;
                        padding: 40px;
                        border-radius: 15px;
                        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
                        text-align: center;
                        max-width: 600px;
                    }}
                    h1 {{ color: #ff6b6b; margin-bottom: 20px; }}
                    p {{ color: #666; margin-bottom: 15px; text-align: left; }}
                    .error-detail {{ 
                        background: #f8f9fa; 
                        padding: 15px; 
                        border-radius: 5px;
                        font-family: monospace;
                        font-size: 12px;
                        text-align: left;
                        margin: 20px 0;
                    }}
                    a {{
                        display: inline-block;
                        background: #667eea;
                        color: white;
                        padding: 12px 30px;
                        text-decoration: none;
                        border-radius: 8px;
                        font-weight: bold;
                        margin-top: 20px;
                    }}
                    a:hover {{ background: #764ba2; }}
                </style>
            </head>
            <body>
                <div class="error-box">
                    <h1>⚠️ Error Occurred</h1>
                    <p><strong>Error message:</strong></p>
                    <div class="error-detail">{str(e)}</div>
                    <p>Please check your input and try again. If the problem persists, contact support.</p>
                    <a href="/">🏠 Go Back Home</a>
                </div>
            </body>
        </html>
        """

@app.route('/health')
def health():
    return "OK", 200

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Flask development server...")
    print("📍 Access at: http://127.0.0.1:5000")
    print("="*60 + "\n")
    app.run(debug=True)