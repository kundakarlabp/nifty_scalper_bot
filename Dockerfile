# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Set unbuffered output for logs
ENV PYTHONUNBUFFERED=1

# Set PYTHONPATH so src/ becomes importable
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first (for cache optimization)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir numpy pandas && \
    pip install --no-cache-dir -r requirements.txt

# Copy full source code
COPY . .

# Make manage_bot.sh executable
RUN chmod +x manage_bot.sh

# Default command: run both real trading + telegram command listener
CMD ["bash", "manage_bot.sh", "start"]
