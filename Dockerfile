# Dockerfile
FROM python:3.11-slim

# Speed up pip & avoid cache bloat
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Upgrade pip tooling first (important for manylinux wheels)
RUN python -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better layer caching
COPY requirements.txt /app/requirements.txt

# Prefer prebuilt wheels; no OS build deps needed
RUN pip install --prefer-binary -r requirements.txt

# Copy the rest of the source
COPY . /app

# Default command (Railway worker or web both fine)
CMD ["bash", "manage_bot.sh", "run"]
