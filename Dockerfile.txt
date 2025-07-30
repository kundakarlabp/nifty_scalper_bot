# Use a clean, official Python base
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies for TA-Lib-bin and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python to output everything (helpful for Railway logs)
ENV PYTHONUNBUFFERED=1

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project into container
COPY . .

# Set PYTHONPATH for src imports
ENV PYTHONPATH=/app/src

# Default start command
CMD ["python", "src/main.py", "--mode", "realtime", "--trade"]
