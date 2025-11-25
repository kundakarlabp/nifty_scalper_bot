# === STAGE 1: BUILDER (Uses Bullseye for stability and OpenSSL 1.1) ===
FROM python:3.11-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# FIX: Force stable mirror (Google CDN) to prevent network failure during apt-get
# This command replaces the default Debian mirror sources list.
RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list

# Install essential build tools, including linear algebra libs for Numpy/Scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install

# Copy the new, pinned requirements file
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# === STAGE 2: RUNTIME (The clean, small execution image) ===
FROM python:3.11-slim-bullseye

# Runtime environment variables
ENV APP_MODULE="nifty_scalper_bot.main" \
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

# Download Zerodha instruments CSV with retry logic
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

# Final command
CMD ["python", "-m", "nifty_scalper_bot.main"]
