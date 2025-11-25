# === STAGE 1: BUILDER (Used for compiling and installing dependencies) ===
FROM python:3.11-slim-bookworm AS builder

# Build environment variables
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

# Install security and build dependencies
# Note: All package names are on a single line, or one package per line with the continuation mark (\)
RUN apt-get update && apt-get install -y --no-install-recommends \
    apt-transport-https ca-certificates build-essential curl git tzdata \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# === STAGE 2: Runtime Image ===
FROM python:3.11-slim-bookworm

# Runtime environment
ENV APP_MODULE="nifty_scalper_bot.main" \
    APP_CMD="python -m ${APP_MODULE}" \
    INSTRUMENTS_CSV_PATH=/app/instruments.csv \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages \
    TZ=Asia/Kolkata

# Install only runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . /app

# Download instruments CSV (with retry logic)
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done

# Health check (if your app exposes an endpoint on port 8000)
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000
# Use exec form to ensure proper signal handling
CMD ["sh", "-c", "python -m ${APP_MODULE}"]
