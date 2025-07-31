# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Unbuffered output for live logs
ENV PYTHONUNBUFFERED=1

# Set PYTHONPATH to allow imports from src/
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to use Docker layer cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy pandas && \
    pip install --no-cache-dir -r requirements.txt

# Copy all application files
COPY . .

# Final command to run in Railway: Real-time trading ENABLED
CMD ["python", "src/main.py", "--mode", "realtime", "--trade"]
