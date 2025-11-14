FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl tzdata build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt requirements_full.txt /app/
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# Copy code
COPY . /app

# Build-time import guard (fail fast if anything is missing)
RUN python scripts/verify_runtime.py

# Default app module; override via env if different
ENV APP_MODULE="nifty_scalper_bot.main"
ENV APP_CMD="python -m ${APP_MODULE}"

# Download Zerodha instruments master during build so it ships with the image
# Ref: https://api.kite.trade/instruments (public CSV)
RUN curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments
ENV INSTRUMENTS_CSV_PATH=/app/instruments.csv

EXPOSE 8000
CMD ["/bin/sh", "-lc", "$APP_CMD"]
