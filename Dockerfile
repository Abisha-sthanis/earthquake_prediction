 # Use official Python 3.9 image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port (Render uses 10000 by default, can also use 8080)
EXPOSE 10000

# Command to run the app with Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:10000", "app:app", "--workers", "1", "--threads", "2"]
