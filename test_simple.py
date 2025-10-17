print("=== SIMPLE TEST ===")
import os
print(f"Directory: {os.getcwd()}")
print(f"Model exists: {os.path.exists('earthquake_lstm_model.keras')}")
print(f"Scaler exists: {os.path.exists('scaler.pkl')}")