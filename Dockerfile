# Simple single-stage build - no complexity
FROM python:3.11-alpine

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TZ=Asia/Kolkata

# Install system dependencies
RUN apk update && apk add --no-cache \
    build-base \
    python3-dev \
    libffi-dev \
    openssl-dev \
    musl-dev \
    linux-headers \
    curl \
    tzdata \
    && rm -rf /var/cache/apk/*

WORKDIR /app

# Copy entire source code first
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Install the package itself in development mode
RUN pip install --no-cache-dir -e .

# Verify imports work
RUN python -c "import nifty_scalper_bot; print('✅ nifty_scalper_bot OK')"
RUN python -c "from nifty_scalper_bot.main import app; print('✅ app OK')"

# Run the application
CMD ["python", "-m", "nifty_scalper_bot.main"]
