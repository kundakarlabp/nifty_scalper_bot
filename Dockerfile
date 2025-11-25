FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Set working directory
WORKDIR /app

# Install system dependencies (minimal set)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy application files first (faster layer caching)
COPY . /app

# Upgrade pip and install requirements in one step
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download Zerodha instruments CSV with fallback
RUN curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments || \
    echo "Warning: Could not download instruments.csv - will retry at runtime"

# Expose port
EXPOSE 8000

# Start the bot
CMD ["python", "-m", "nifty_scalper_bot.main"]
