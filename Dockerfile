FROM python:3.10-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python packages (no C compilation needed)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Expose port
EXPOSE 10000

# Start the app
CMD ["python", "nifty_scalper_bot.py"]
