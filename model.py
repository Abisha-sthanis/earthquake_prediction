# model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Input
import joblib

# ============================================
# 1. LOAD DATA
# ============================================
earthquake = pd.read_csv('E:\earthquakeprediction\Istanbul Deprem.csv', encoding='ISO-8859-9', sep='\t')

earthquake['Date'] = pd.to_datetime(earthquake['Date'], format='%d-%m-%Y %H:%M', errors='coerce')
earthquake['date'] = earthquake['Date'].dt.day
earthquake['month'] = earthquake['Date'].dt.month
earthquake['year'] = earthquake['Date'].dt.year
earthquake['hour'] = earthquake['Date'].dt.hour
earthquake['minute'] = earthquake['Date'].dt.minute
earthquake['second'] = earthquake['Date'].dt.second

earthquake = earthquake.dropna()

# Encode categorical feature (Type)
le = LabelEncoder()
earthquake['Type'] = le.fit_transform(earthquake['Type'])

# Features and target
X = earthquake.drop(columns=['Magnitude', 'Date', 'Location', 'EventID'])
Y = earthquake['Magnitude']

# Split before scaling (✅ avoids data leakage)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Scale data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM
X_train_scaled = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_scaled = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build LSTM model
n_features = X_train.shape[1]
model = Sequential([
    Input(shape=(1, n_features)),
    Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='linear')
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.fit(X_train_scaled, Y_train, validation_split=0.2, epochs=50, batch_size=32, verbose=1)

# Save model and scalers
model.save('earthquake_lstm_model.keras')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(le, 'label_encoder.pkl')

print("✅ Model, scaler, and encoder saved successfully!")
