# Use official Python slim image
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app  # ✅ FIXED HERE

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libffi-dev \
    libssl-dev \
    curl \
    ca-certificates \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

RUN if [ -f manage_bot.sh ]; then chmod +x manage_bot.sh; fi

CMD ["python3", "src/main.py", "start"]  # ✅ FIXED HERE
