#!/bin/bash

echo "🔁 Checking environment..."
if [[ -z "$ZERODHA_API_KEY" || -z "$ZERODHA_ACCESS_TOKEN" ]]; then
  echo "❌ ERROR: ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN is not set in environment."
  exit 1
fi

echo "✅ Environment OK. Starting Python script..."

python3 <<EOF
import os
import pandas as pd
from kiteconnect import KiteConnect

api_key = os.environ.get("ZERODHA_API_KEY")
access_token = os.environ.get("ZERODHA_ACCESS_TOKEN")

kite = KiteConnect(api_key=api_key)
kite.set_access_token(access_token)

print("🔍 Fetching ALL NFO instruments...")
instruments = kite.instruments("NFO")
df = pd.DataFrame(instruments)

# Save full file first (for debugging)
df.to_csv("nfo_instruments_full.csv", index=False)
print("✅ Saved full NFO instrument list to: nfo_instruments_full.csv")

# Try filtering for NIFTY weekly options
nifty_opts = df[(df['name'] == 'NIFTY') & (df['instrument_type'].isin(['CE', 'PE']))]
nifty_opts.to_csv("nifty_weekly_options.csv", index=False)
print("✅ Saved: nifty_weekly_options.csv")

# Check for specific strike
target_symbol = "NIFTY25AUG24550CE"
match = nifty_opts[nifty_opts['tradingsymbol'] == target_symbol]

if not match.empty:
    print(f"✅ Symbol found: {target_symbol}")
else:
    print(f"❌ Symbol NOT found: {target_symbol}")
EOF
