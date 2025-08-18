# Use official Python slim image
FROM python:3.10-slim

# --- Security: non-root user ---
ARG UID=10001
RUN adduser \
    --disabled-password \
    --gecos "" \
    --home "/nonexistent" \
    --shell "/sbin/nologin" \
    --no-create-home \
    --uid "${UID}" \
    appuser

# Workdir
WORKDIR /app

# Logs unbuffered
ENV PYTHONUNBUFFERED=1

# ✅ Make both /app and /app/src importable everywhere
ENV PYTHONPATH="/app:/app/src"

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libffi-dev libssl-dev curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# App source
COPY . .

# Ensure .sh is executable before switching user
RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

# Use non-root
USER appuser

# ✅ Run module with explicit 'start' arg so the process keeps running
CMD ["python3", "-m", "src.main", "start"]
