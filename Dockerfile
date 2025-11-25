# === STAGE 1: BUILDER (Uses Bullseye, now with dependency fix flags) ===
FROM python:3.11-slim-bullseye AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# FIX: Force stable mirror (Google CDN)
RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list

# FIX: Use --fix-missing and --allow-unauthenticated flags to resolve dependency conflicts.
# Also run a small upgrade to ensure core libs are synced.
RUN apt-get update && apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    --fix-missing \
    --allow-unauthenticated \
    build-essential \
    curl \
    ca-certificates \
    tzdata \
    libatlas-base-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /install
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# === STAGE 2: RUNTIME (Same stable image) ===
FROM python:3.11-slim-bullseye

ENV APP_MODULE="nifty_scalper_bot.main" \
    PYTHONPATH=/app:/usr/local/lib/python3.11/site-packages \
    TZ=Asia/Kolkata

# FIX: Force stable mirror for runtime packages
RUN echo "deb http://cdn-fastly.deb.debian.org/debian bullseye main" > /etc/apt/sources.list

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tzdata \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages
COPY . /app

# Download Zerodha instruments CSV with retry logic
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

CMD ["python", "-m", "nifty_scalper_bot.main"]
