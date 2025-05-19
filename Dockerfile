FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements_keras.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements_keras.txt

# Copy application code
COPY appKeras.py .
COPY templates/ ./templates/
COPY static/ ./static/

# Create uploads directory
RUN mkdir -p ./static/uploads
RUN mkdir -p ./static/models

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=2

# Expose port
EXPOSE 5000

# Run application
CMD ["python", "appKeras.py"] 