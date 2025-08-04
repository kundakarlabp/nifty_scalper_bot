# Use official Python slim image
FROM python:3.10-slim

# Set non-root user (security best practice)
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Set working directory
WORKDIR /app

# Ensure unbuffered logs
ENV PYTHONUNBUFFERED=1

# Set PYTHONPATH
ENV PYTHONPATH=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Set permissions
USER appuser
RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

# Run using module for correct imports
CMD ["python3", "-m", "src.main"]
