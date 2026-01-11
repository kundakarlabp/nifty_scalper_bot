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

# Install Python dependencies
COPY requirements.txt pyproject.toml setup.py* ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy ALL source code
COPY . .

# Install package
RUN pip install --no-cache-dir .

# ============================================================================
# STAGE 2 — RUNTIME
# ============================================================================
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Install runtime tools (bash for entrypoint, curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata curl bash ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependencies and code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

# --- ENTRYPOINT SETUP ---
COPY entrypoint.sh /app/entrypoint.sh
# Fix Windows Line Endings (CRITICAL)
RUN sed -i 's/\r$//' /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Launch via Entrypoint
CMD ["/app/entrypoint.sh"]
