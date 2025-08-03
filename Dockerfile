# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Ensure unbuffered logs for real-time output
ENV PYTHONUNBUFFERED=1

# Set PYTHONPATH so absolute imports like 'from src.xyz' work
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files into the container
COPY . .

# Make manage script executable (if used)
RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

# ✅ Run the main bot script directly
CMD ["python3", "src/main.py"]