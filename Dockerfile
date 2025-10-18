# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

# Install system dependencies for TensorFlow and HDF5
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements file FIRST (for better caching)
COPY requirements.txt .

# Install dependencies with increased timeout
RUN pip install --no-cache-dir --upgrade pip --timeout 1000 \
    && pip install --no-cache-dir -r requirements.txt --timeout 1000

# Copy ALL application files (including templates folder and model files)
COPY . .

# Verify critical files exist
RUN ls -la && \
    test -f app.py && echo "✅ app.py found" || echo "❌ app.py missing" && \
    test -f earthquake_lstm_model.keras && echo "✅ Model found" || echo "❌ Model missing" && \
    test -f scaler.pkl && echo "✅ Scaler found" || echo "❌ Scaler missing" && \
    test -d templates && echo "✅ templates/ found" || echo "❌ templates/ missing"

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Command to run the app with Gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-10000} app:app --workers 1 --threads 2 --timeout 120 --log-level info --access-logfile - --error-logfile -