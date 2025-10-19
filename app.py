from flask import Flask, render_template, request
import numpy as np
import os
import traceback

app = Flask(__name__)

# ============================================
# ADD THIS DIAGNOSTIC SECTION (NEW)
# ============================================
print("\n" + "="*60)
print("🔍 DIAGNOSTIC INFORMATION")
print("="*60)
print(f"Working Directory: {os.getcwd()}")
print(f"\nAll files in current directory:")
for item in sorted(os.listdir('.')):
    file_path = os.path.join('.', item)
    if os.path.isfile(file_path):
        size = os.path.getsize(file_path)
        print(f"  📄 {item} ({size:,} bytes)")
    else:
        print(f"  📁 {item}/")
print(f"\n🔍 Checking critical files:")
print(f"  earthquake_lstm_model.keras exists: {os.path.exists('earthquake_lstm_model.keras')}")
print(f"  scaler.pkl exists: {os.path.exists('scaler.pkl')}")
print("="*60 + "\n")
# ============================================
# END OF DIAGNOSTIC SECTION
# ============================================

# Initialize model and scaler as None
model = None
scaler = None

# Try to load model and scaler
try:
    from tensorflow.keras.models import load_model
    import joblib
    
    if os.path.exists('earthquake_lstm_model.keras') and os.path.exists('scaler.pkl'):
        model = load_model('earthquake_lstm_model.keras')
        scaler = joblib.load('scaler.pkl')
        print("✅ Model and scaler loaded successfully!")
    else:
        print("⚠️ Warning: Model files not found. Prediction will not work.")

except Exception as e:
    print(f"⚠️ Error loading model: {e}")
    traceback.print_exc()

# ALL YOUR EXISTING ROUTES STAY EXACTLY THE SAME
@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['POST'])
def input_page():
    name = request.form.get('name', 'User')
    return render_template('input.html', name=name)

@app.route('/predict', methods=['POST'])
def predict():
    # ... your existing predict code stays here ...
    pass

# ... rest of your code stays the same ...