Project Title:   Earthquake Magnitude Prediction System

Short Description
 A machine learning–powered web application that predicts earthquake magnitudes using an LSTM (Long Short-Term Memory) neural network. The system takes seismic features (geographic coordinates, 
depth, RMS values, temporal data) and forecasts the probable magnitude and associated severity level.

Key Features:

Real-time magnitude prediction with severity classification (e.g. Minor → Major/Great)

Interactive web interface built with Flask: form inputs, instant result display

Color-coded alerts (🟢 Minor → 🔴 Major/Great)

Preprocessing and scaling of input features

Model persistence (with joblib or saving Keras model) for deployment

Backend stack: Python, Flask, TensorFlow / Keras, NumPy

Frontend: HTML5, CSS3, JavaScript

Use Case / Motivation:
This system can support seismologists, researchers, and disaster management teams by providing early estimates of earthquake magnitude to aid in risk assessment, resource planning, and alert escalation.

How It Works (Architecture / Flow Summary)

User (or upstream system) inputs seismic parameters (latitude, longitude, depth, RMS, timestamp, etc.)

Inputs are scaled / normalized and fed into the trained LSTM model

Model predicts a magnitude value

A classification / severity engine maps the magnitude to categories (e.g. minor, moderate, major)

The UI displays the result with color coding and possibly warnings / recommendations

Limitations & Considerations

Earthquake prediction is inherently uncertain and many geophysical variables are not fully observable

Rare large magnitude events mean class imbalance — model may be biased toward lower magnitudes

Input data latency, noise, missing values must be handled carefully (imputation, validation)

False positives or negatives can have serious consequences
