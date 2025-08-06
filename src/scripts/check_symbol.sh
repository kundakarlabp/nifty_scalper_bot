#!/bin/bash
echo "🔁 Checking environment..."
if [[ -z "$ZERODHA_API_KEY" || -z "$ZERODHA_ACCESS_TOKEN" ]]; then
  echo "❌ ERROR: Missing environment variables."
  exit 1
fi
python3 <<EOF
import os
import pandas as pd
from kiteconnect import KiteConnect

kite = KiteConnect(api_key=os.getenv("ZERODHA_API_KEY"))
kite.set_access_token(os.getenv("ZERODHA_ACCESS_TOKEN"))
print("🔍 Fetching NFO instruments...")
df = pd.DataFrame(kite.instruments("NFO"))
df.to_csv("nfo_full.csv", index=False)
matches = df[df['tradingsymbol'] == "NIFTY25AUG24550CE"]
print("✅ Match found!" if not matches.empty else "❌ No match for NIFTY25AUG24550CE")
EOF
