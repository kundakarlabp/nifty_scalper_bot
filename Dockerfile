# ============================================================================
# STAGE 1 — BUILDER
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc python3-dev libffi-dev libssl-dev curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies
COPY requirements.txt pyproject.toml setup.py* ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy source and install package
COPY . .
RUN pip install --no-cache-dir .

# ============================================================================
# STAGE 2 — RUNTIME
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Install runtime tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- CRITICAL FIXES HERE ---
# 1. Copy Python Libraries
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
# 2. Copy Executables (Fixes 'uvicorn not found')
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy App Code
COPY --from=builder /app /app

# Setup Entrypoint
COPY entrypoint.sh /app/entrypoint.sh
# Remove Windows carriage returns just in case
RUN sed -i 's/\r$//' /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Launch
CMD ["/app/entrypoint.sh"]
