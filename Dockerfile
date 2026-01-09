# Multi-stage build for production - optimized for Railway.app
FROM python:3.11-alpine AS builder

# ===== STAGE 1: Build Stage =====
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build dependencies only in builder stage
RUN apk add --no-cache --virtual .build-deps \
    build-base \
    python3-dev \
    libffi-dev \
    openssl-dev \
    musl-dev \
    linux-headers \
    gcc

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY requirements.txt pyproject.toml setup.py* ./
COPY src ./src

# Install Python dependencies with strict error checking
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt && \
    pip install -e . && \
    pip check

# Verify critical imports at build time
RUN python -c "import nifty_scalper_bot; print('✅ nifty_scalper_bot imported')" && \
    python -c "from nifty_scalper_bot.main import app; print('✅ app imported')"

# ===== STAGE 2: Runtime Stage =====
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata \
    PYTHONPATH=/app/src:$PYTHONPATH

# Install only runtime dependencies (smaller image)
RUN apk add --no-cache \
    tzdata \
    curl \
    ca-certificates

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code from builder
COPY --from=builder /app/src ./src
COPY --from=builder /app/pyproject.toml .

# Health check - Railway uses this to verify readiness
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Use exec form to properly handle signals (SIGTERM)
ENTRYPOINT ["/usr/local/bin/python"]
CMD ["-m", "nifty_scalper_bot.main"]
