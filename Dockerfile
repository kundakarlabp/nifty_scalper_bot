FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TZ=Asia/Kolkata
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ \
  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN python -m pip install --upgrade pip && pip install -r requirements.txt

COPY . /app
# make sure default config is inside the image
RUN mkdir -p /app/config && cp -n config/strategy.yaml /app/config/strategy.yaml
RUN chmod +x manage_bot.sh run.sh

CMD ["./run.sh"]
