# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Unbuffered output ensures logs are flushed immediately
ENV PYTHONUNBUFFERED=1

# Add src to PYTHONPATH so internal packages resolve correctly
ENV PYTHONPATH=/app/src

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the codebase
COPY . .

# Ensure manage_bot.sh is executable if present
RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

# Default command runs the main entry point
CMD ["python3", "-m", "src.main", "start"]