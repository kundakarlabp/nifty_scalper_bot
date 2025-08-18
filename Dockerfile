# Use slim python with build tools for numpy/pandas
FROM python:3.11-slim

# System deps for numpy/pandas/scipy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libatlas-base-dev liblapack-dev libblas-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . /app

# Default env
ENV PYTHONUNBUFFERED=1

# Railway/Heroku style: PORT provided; our health server reads it
CMD ["python", "-m", "src.main", "start"]
