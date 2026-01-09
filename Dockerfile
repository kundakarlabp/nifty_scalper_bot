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

ENV APP_MODULE="nifty_scalper_bot.main" \
    TZ=Asia/Kolkata

RUN apk update && apk add --no-cache \
    curl \
    tzdata \
    libstdc++ \
    openblas \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy installed dependencies
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy application code
COPY . /app

# ✅ REQUIRED FOR editable install WITH pyproject.toml
RUN pip install --upgrade pip setuptools wheel

# ✅ INSTALL YOUR PACKAGE (NOW IT ACTUALLY INSTALLS)
RUN pip install --no-cache-dir -e .

# Download Zerodha instruments CSV
RUN for i in 1 2 3; do \
        curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments && break || sleep 5; \
    done || true

CMD ["python", "-m", "nifty_scalper_bot.main"]
