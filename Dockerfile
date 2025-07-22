# Use official Python 3.10 image (compatible with your .python-version)
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies for TA-Lib (this is the key step!)
RUN apt-get update && apt-get install -y \
    build-essential \
    libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt first
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Start the app
CMD ["python", "nifty_scalper_bot.py"]
