# Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies for TensorFlow and HDF5
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install dependencies with increased timeout
RUN pip install --no-cache-dir --upgrade pip --timeout 1000 \
    && pip install --no-cache-dir -r requirements.txt --timeout 1000

# Copy the rest of the app
COPY . .

# Expose port (Render uses PORT env variable)
EXPOSE 10000

# Command to run the app with Gunicorn
CMD gunicorn --bind 0.0.0.0:$PORT app:app --workers 1 --threads 2 --timeout 120 --log-level info