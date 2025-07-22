# Use a stable Python image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib C library (fix for compilation errors)
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Set library path
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements
COPY requirements.txt .

# Upgrade pip and install Python packages
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 10000

# Start the app
CMD ["python", "nifty_scalper_bot.py"]
