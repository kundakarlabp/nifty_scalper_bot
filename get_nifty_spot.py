import os
from kiteconnect import KiteConnect

# ✅ Manual .env loader
def load_env(file_path=".env"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"{file_path} not found")
    
    with open(file_path) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, value = line.strip().split("=", 1)
                os.environ[key] = value

# Load .env manually
load_env()

# Get API credentials from environment
api_key = os.environ["ZERODHA_API_KEY"]
access_token = os.environ["ZERODHA_ACCESS_TOKEN"]

# Initialize Kite client
kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

# Fetch NIFTY 50 Spot data
try:
    quote = kite.quote("NSE:NIFTY 50")
    print("✅ NIFTY 50 Spot Price:", quote["NSE:NIFTY 50"]["last_price"])
except Exception as e:
    print("❌ Error:", e)
