import tensorflow as tf
from tensorflow.keras import models
import joblib

print("Loading model...")
model = models.load_model('earthquake_lstm_model.keras')

print("Re-saving model with TensorFlow 2.15.0...")
model.save('earthquake_lstm_model.keras')

print("Loading scaler...")
scaler = joblib.load('scaler.pkl')

print("Re-saving scaler...")
joblib.dump(scaler, 'scaler.pkl')

print("✅ Model and scaler re-saved successfully!")