# Use Python 3.10 slim image
FROM python:3.10-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y gcc g++ && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Correct installation sequence
RUN pip install --upgrade pip
RUN pip install "setuptools>=61.0" wheel
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN useradd --create-home --shell /bin/bash app && chown -R app:app /app
USER app

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

CMD ["python", "nifty_scalper_bot.py"]
