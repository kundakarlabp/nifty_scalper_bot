# ========================================================================
# STAGE 1: BUILDER
# ========================================================================
FROM python:3.11-alpine AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

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

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install/deps

# ========================================================================
# STAGE 2: RUNTIME
# ========================================================================
FROM python:3.11-alpine

ENV TZ=Asia/Kolkata

RUN apk update && apk add --no-cache \
    curl \
    tzdata \
    libstdc++ \
    openblas \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy Python dependencies
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy source code
COPY . /app

# 🔴 CRITICAL: install build tooling for pyproject.toml
RUN pip install --upgrade pip setuptools wheel

# 🔴 CRITICAL: install YOUR package
RUN pip install --no-cache-dir

# Optional: download instruments
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

CMD ["python", "-m", "nifty_scalper_bot.main"]
