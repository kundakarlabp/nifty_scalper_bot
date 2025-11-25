# ========================================================================
# STAGE 1: BUILDER (Alpine for ultimate network stability & small size)
# ========================================================================
FROM python:3.11-alpine AS builder

# Set crucial environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install essential Alpine build tools (required for Numpy/Pandas/Scipy)
# This uses apk, which is more reliable than apt-get on flaky networks.
RUN apk update && apk add --no-cache \
    build-base \
    python3-dev \
    libffi-dev \
    openssl-dev \
    musl-dev \
    linux-headers \
    curl \
    tzdata \
    openblas-dev \
    && rm -rf /var/cache/apk/*

WORKDIR /install

# Copy the pinned requirements file (CRUCIAL: ensure requirements.txt is pinned!)
COPY requirements.txt .
# Install dependencies, prioritizing minimal memory usage
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# ========================================================================
# STAGE 2: RUNTIME (The minimal Alpine execution image)
# ========================================================================
FROM python:3.11-alpine

# Runtime environment variables
ENV APP_MODULE="nifty_scalper_bot.main" \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages \
    TZ=Asia/Kolkata

# Install runtime dependencies only
RUN apk update && apk add --no-cache \
    curl \
    tzdata \
    libstdc++ \
    openblas \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy installed packages from the builder stage
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . /app

# Download Zerodha instruments CSV with retry logic
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

# Final command
CMD ["python", "-m", "nifty_scalper_bot.main"]
