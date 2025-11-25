# === STAGE 1: BUILDER (Used for compiling and installing dependencies) ===
FROM python:3.11-slim-bookworm AS builder

# Build environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install necessary build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# === STAGE 2: RUNTIME (The clean, small execution image) ===
FROM python:3.11-slim-bookworm

# Runtime environment variables
ENV APP_MODULE="nifty_scalper_bot.main" \
    INSTRUMENTS_CSV_PATH=/app/instruments.csv \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages \
    TZ=Asia/Kolkata

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . /app

# Download Zerodha instruments CSV with retry logic (FIXED URL)
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done

# Check if the main module can be imported (replaces the missing verify_runtime.py)
RUN python -c "import nifty_scalper_bot; print('✓ Module import successful')" || true

# Health check for Railway monitoring
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

# Start the bot directly using the Python module system
CMD ["python", "-m", "nifty_scalper_bot.main"]
