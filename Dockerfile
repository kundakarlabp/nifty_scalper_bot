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

# Install runtime dependencies
RUN apk update && apk add --no-cache \
    curl \
    tzdata \
    libffi \
    openssl \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy installed packages from builder stage
COPY --from=builder /install /usr/local/lib/python3.11/site-packages/

# Copy source code
COPY . /app

# ===================================================================
# Install the package itself
# ===================================================================
RUN pip install --no-cache-dir -e . && \
    python -c "from nifty_scalper_bot.main import app; print('✅ Package verified')"

# ===================================================================
# ENTRYPOINT - Run the bot
# ===================================================================
CMD ["python", "-m", "nifty_scalper_bot.main"]
