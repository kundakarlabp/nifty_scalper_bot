# Use official Python slim image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Ensure logs are unbuffered for real-time output
ENV PYTHONUNBUFFERED=1

# ✅ Set PYTHONPATH so absolute imports like `from src.xyz` work
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy only requirements file first for Docker cache optimization
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Make the manage script executable (if used)
RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

# ✅ Default command to run your bot
CMD ["python3", "src/main.py", "start"]
