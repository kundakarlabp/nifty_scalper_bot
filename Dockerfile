# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py* ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install the package
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

# Set working directory
WORKDIR /app

# ============================================================
# FIX: Create data directories with proper permissions BEFORE
# switching to non-root user
# ============================================================
RUN mkdir -p /app/data /tmp/nifty_scalper_data \
    && chmod 777 /app/data /tmp/nifty_scalper_data

# Copy and setup entrypoint
COPY entrypoint.sh /app/entrypoint.sh
RUN sed -i 's/\r$//' /app/entrypoint.sh && chmod +x /app/entrypoint.sh

# Set DATA_DIR environment variable as fallback
ENV DATA_DIR=/tmp/nifty_scalper_data

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Create non-root user for security
# NOTE: /app/data and /tmp/nifty_scalper_data are already chmod 777
# so the appuser can write to them
RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app /tmp/nifty_scalper_data

USER appuser

# Use entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]
