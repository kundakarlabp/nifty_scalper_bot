# syntax=docker/dockerfile:1

############################
# Stage 1: Builder
############################
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System build deps (removed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --upgrade pip

# Copy requirements and build wheels
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt


############################
# Stage 2: Final runtime
############################
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=Asia/Kolkata

WORKDIR /app

# Runtime libs only (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 tzdata \
    && rm -rf /var/lib/apt/lists/* \
    && ln -snf /usr/share/zoneinfo/$TZ /etc/localtime \
    && echo $TZ > /etc/timezone

# Install wheels built in builder
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/* && rm -rf /wheels

# Copy application code
COPY . .

# Create non-root user
RUN groupadd -r app && useradd -r -g app app \
    && chown -R app:app /app
USER app

# Optional healthcheck if you expose Flask/FastAPI health endpoint
# HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
#   CMD curl -f http://localhost:8000/health || exit 1

# Default command: run the bot
CMD ["python", "-m", "src.main", "start"]
