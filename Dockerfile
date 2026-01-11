# ============================================================================
# STAGE 1 — BUILDER
# ============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions
COPY requirements.txt pyproject.toml setup.py* ./

# Install deps
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy ALL code (Fixes the 'src' error)
COPY . .

# Install your package
RUN pip install --no-cache-dir .

# ============================================================================
# STAGE 2 — RUNTIME
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Install runtime tools (bash is required for entrypoint.sh)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy libraries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages

# Copy App Code
COPY --from=builder /app /app

# --- INSTALL ENTRYPOINT (WITH WINDOWS FIX) ---
COPY entrypoint.sh /app/entrypoint.sh

# 🪄 MAGIC LINE: Removes Windows \r characters so script runs on Linux
RUN sed -i 's/\r$//' /app/entrypoint.sh

RUN chmod +x /app/entrypoint.sh

# Start using the diagnostic script
CMD ["/app/entrypoint.sh"]
