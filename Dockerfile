# === STAGE 1: BUILDER (Used for compiling and installing dependencies) ===
FROM python:3.11-alpine AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install build tools and other necessary system deps
RUN apk update && apk add --no-cache \
    build-base \
    tzdata \
    curl \
    python3-dev \
 && rm -rf /var/cache/apk/*

WORKDIR /install

# Copy requirements and install dependencies into a separate directory
# '--target /install/deps' ensures packages are isolated for easy copying
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --target /install/deps


# === STAGE 2: FINAL (The minimal image for execution) ===
# Use the same base image, but without the build tools
FROM python:3.11-alpine

# Set environment variables for runtime
ENV APP_MODULE="nifty_scalper_bot.main"
ENV APP_CMD="python -m ${APP_MODULE}"
ENV INSTRUMENTS_CSV_PATH=/app/instruments.csv

# Install runtime-only dependencies (like curl and tzdata)
RUN apk update && apk add --no-cache \
    tzdata \
    curl \
 && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy ONLY the installed Python packages from the builder stage
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages

# Copy code and run build-time checks
COPY . /app
RUN python scripts/verify_runtime.py

# Download Zerodha instruments master
RUN curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments

EXPOSE 8000
CMD ["/bin/sh", "-lc", "$APP_CMD"]
