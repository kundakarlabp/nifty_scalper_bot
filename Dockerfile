# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set unbuffered output (for better Railway/Render log visibility)
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set PYTHONPATH so src/ becomes importable
ENV PYTHONPATH=/app/src

# Copy requirements file first (to leverage Docker layer cache)
COPY requirements.txt .

# Install dependencies in correct order
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy pandas && \
    pip install --no-cache-dir -r requirements.txt

# Copy full application code into the container
COPY . .

# Default command to run your bot
CMD ["python", "src/main.py", "--mode", "realtime", "--trade"]
