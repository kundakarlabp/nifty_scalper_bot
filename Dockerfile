# ============================================================================
# STAGE 1 — BUILDER
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency metadata
COPY requirements.txt pyproject.toml setup.py* ./

# Install dependencies & the package itself
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir .

# Sanity Check: Ensure imports work during build
RUN python -c "import nifty_scalper_bot; print('✅ nifty_scalper_bot imported')" \
 && python -c "from nifty_scalper_bot.main import app; print('✅ app imported')"

# ============================================================================
# STAGE 2 — RUNTIME (The Production Image)
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Runtime OS dependencies (bash/curl needed for debugging/healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Copy source code (optional if installed as package, but good for reference)
COPY --from=builder /app/src ./src

# --- CRITICAL: Copy and Setup Entrypoint ---
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port (Documentation only, Railway ignores this)
EXPOSE 8000

# Use the diagnostic entrypoint
CMD ["/app/entrypoint.sh"]
