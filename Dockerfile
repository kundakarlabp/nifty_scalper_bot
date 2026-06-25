# File purpose: Build the production image for the release-verified trading service.
# Key responsibilities: Install dependencies, embed the Railway commit SHA, run health checks, and start the verified ASGI entrypoint.
# Operational constraints: Preserve the embedded revision and do not bypass deployment_main.py or /releasez.

# Multi-stage build for optimized production image
FROM python:3.11-slim as builder

ARG RAILWAY_GIT_COMMIT_SHA=unknown

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt pyproject.toml setup.py* ./
RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

# Copy the exact source tree and persist the Railway source revision inside the image.
COPY . .
RUN printf '%s\n' "${RAILWAY_GIT_COMMIT_SHA}" > /app/.build_commit_sha

# Install the package
RUN pip install --no-cache-dir .

# Production stage
FROM python:3.11-slim

ARG RAILWAY_GIT_COMMIT_SHA=unknown
ENV APP_BUILD_SHA=${RAILWAY_GIT_COMMIT_SHA}
LABEL org.opencontainers.image.revision=${RAILWAY_GIT_COMMIT_SHA}

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    curl \
    bash \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the installed environment and the same verified source/build marker.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

WORKDIR /app

RUN mkdir -p /app/data \
    && chmod 777 /app/data

# Local/container runtimes use the same default port as the ASGI command.
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/releasez || exit 1

RUN groupadd -r appuser && useradd -r -g appuser appuser \
    && chown -R appuser:appuser /app

USER appuser

CMD ["sh", "-c", "python -m uvicorn nifty_scalper_bot.deployment_main:app --host 0.0.0.0 --port ${PORT:-8080}"]
