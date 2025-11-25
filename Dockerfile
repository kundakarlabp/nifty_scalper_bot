# ========================================================================
# STAGE 1: BUILDER (Used for compiling Python packages and dependencies)
# ========================================================================
# Uses Bullseye for crucial libssl.so.1.1 and glibc compatibility
FROM python:3.11-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# FIX: Force stable mirror (Google CDN) to prevent network failure during apt-get
RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list

# Install essential build tools (compiler, headers, linear algebra libraries)
# 'build-essential' for cc/gcc. 'libatlas-base-dev' for optimized NumPy/SciPy.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# ========================================================================
# STAGE 2: RUNTIME (The minimal execution image)
# ========================================================================
FROM python:3.11-slim-bullseye

# Runtime environment variables (IST timezone set for Nifty trading)
ENV APP_MODULE="nifty_scalper_bot.main" \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages \
    TZ=Asia/Kolkata

# FIX: Force stable mirror for runtime packages as well
RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list

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

# Download Zerodha instruments CSV with retry logic (robustness fix)
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

# Final command to start the bot
CMD ["python", "-m", "nifty_scalper_bot.main"]
