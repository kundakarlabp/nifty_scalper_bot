# Use a base image with Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /app

# Install system dependencies (this is how we install TA-Lib properly)
RUN apt-get update && apt-get install -y \
    build-essential \
    libta-lib-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose the port Render expects
EXPOSE 10000

# Set the start command
CMD ["python", "nifty_scalper_bot.py"]
