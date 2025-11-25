# 1. Use the stable Alpine base image
FROM python:3.11-alpine

# Environment Variables (mostly preserved)
# PIP_NO_CACHE_DIR, PYTHONDONTWRITEBYTECODE, PYTHONUNBUFFERED are good practice.
ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# 2. Install necessary system dependencies for compiling Python packages and for curl
# 'build-base' is the Alpine equivalent of 'build-essential'
# 'tzdata' is needed for time zone operations (important for trading)
# 'curl' is needed to download the instruments file
RUN apk update && apk add --no-cache \
    build-base \
    tzdata \
    curl \
    python3-dev \
 && rm -rf /var/cache/apk/*

WORKDIR /app

# 3. Install deps first (layer caching)
COPY requirements.txt requirements_full.txt /app/
RUN python -m pip install --upgrade pip \
 && pip install -r /app/requirements.txt

# 4. Copy code
COPY . /app

# 5. Build-time import guard (fail fast if anything is missing)
# Ensure the 'scripts/verify_runtime.py' file is present in your repo
RUN python scripts/verify_runtime.py

# 6. Default app module and command
ENV APP_MODULE="nifty_scalper_bot.main"
# Use 'python -m' which works better across environments
ENV APP_CMD="python -m ${APP_MODULE}"

# 7. Download Zerodha instruments master during build
# Ref: https://api.kite.trade/instruments (public CSV)
RUN curl -fsSL -o /app/instruments.csv https://api.kite.trade/instruments
ENV INSTRUMENTS_CSV_PATH=/app/instruments.csv

# 8. Expose port (if your app uses a web interface or API)
EXPOSE 8000
# CMD starts the application (using shell to interpret the $APP_CMD variable)
CMD ["/bin/sh", "-lc", "$APP_CMD"]
