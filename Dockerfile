# ===================================================================
# STAGE 1: BUILDER - Install dependencies
# ===================================================================
FROM python:3.11-alpine AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apk update && apk add --no-cache \
    build-base \
    python3-dev \
    libffi-dev \
    openssl-dev \
    musl-dev \
    linux-headers \
    curl \
    tzdata

WORKDIR /install

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt --target /install

# ===================================================================
# STAGE 2: RUNTIME - Create final image
# ===================================================================
FROM python:3.11-alpine

ENV TZ=Asia/Kolkata \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install runtime dependencies only
RUN apk update && apk add --no-cache \
    curl \
    tzdata \
    libffi \
    openssl

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install/deps /usr/local/lib/python3.11/site-packages/ || true
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Copy source code
COPY . /app

# ===================================================================
# CRITICAL: Install the package itself in editable mode
# ===================================================================
RUN pip install --no-cache-dir -e . && \
    python -c "from nifty_scalper_bot.main import app; print('✅ Package imported successfully')" || \
    (echo "❌ FAILED TO IMPORT PACKAGE" && exit 1)

# ===================================================================
# Verify everything is installed
# ===================================================================
RUN python -c "import nifty_scalper_bot; print('✅ nifty_scalper_bot module found')" && \
    python -c "from nifty_scalper_bot.core.app import get_http_app; print('✅ get_http_app imported')" && \
    echo "✅ All imports verified!"

# ===================================================================
# Optional: Download instruments (for faster startup)
# ===================================================================
RUN for i in 1 2 3; do \
      curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments || true; \
      [ -f /app/instruments.csv ] && break; \
      echo "Retry $i..."; \
    done || echo "⚠️ Could not download instruments (will fetch at runtime)"

# ===================================================================
# Health check
# ===================================================================
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from nifty_scalper_bot.main import app; print('OK')" || exit 1

# ===================================================================
# ENTRYPOINT - Run the bot
# ===================================================================
CMD ["python", "-m", "nifty_scalper_bot.main"]
