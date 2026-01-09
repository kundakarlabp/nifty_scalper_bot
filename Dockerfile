# ============================================================================
# STAGE 1 — BUILDER
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps needed to build wheels
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

# Copy only dependency metadata first (for Docker cache efficiency)
COPY requirements.txt pyproject.toml setup.py* ./

# Install dependencies into a virtual location
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src

# Install YOUR package (NON-editable — critical)
RUN pip install --no-cache-dir .

# Sanity check: ensure imports work (Railway does this too)
RUN python -c "import nifty_scalper_bot; print('✅ nifty_scalper_bot imported')" \
 && python -c "from nifty_scalper_bot.main import app; print('✅ app imported')"

# ============================================================================
# STAGE 2 — RUNTIME
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Runtime OS deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed Python packages
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Copy application source
COPY --from=builder /app/src ./src
COPY pyproject.toml .

# Expose port (Railway injects $PORT)
EXPOSE 8000

# IMPORTANT:
# DO NOT use python -m ...
# Railway must run uvicorn directly
CMD ["uvicorn", "nifty_scalper_bot.main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
