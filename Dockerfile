# Use official Python 3.10
FROM python:3.10

# Set working directory
WORKDIR /app

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    tar \
    && rm -rf /var/lib/apt/lists/*

# Download and build TA-Lib C library
RUN wget -q http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make -j$(nproc) && \
    make install && \
    cd .. && \
    rm -rf ta-lib ta-lib-0.4.0-src.tar.gz

# Update library cache so 'libta_lib.so' is found
RUN ldconfig

# Set library path (critical for linking)
ENV LD_LIBRARY_PATH /usr/local/lib:$LD_LIBRARY_PATH

# Copy requirements.txt
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
