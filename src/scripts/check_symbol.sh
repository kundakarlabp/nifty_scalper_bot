#!/usr/bin/env bash
set -euo pipefail

# Optional .env
[[ -f .env ]] && source .env

: "${ZERODHA_API_KEY:?ZERODHA_API_KEY not set}"
: "${ZERODHA_ACCESS_TOKEN:?ZERODHA_ACCESS_TOKEN not set}"

SYMBOL="${1:-NIFTY25AUG24550CE}"
OUT_ALL="${OUT_ALL:-nfo_full.csv}"

python3 - <<'PY'
import os, sys, pandas as pd
from kiteconnect import KiteConnect

api_key = os.environ["ZERODHA_API_KEY"]
access = os.environ["ZERODHA_ACCESS_TOKEN"]
symbol  = os.environ.get("SYMBOL", "NIFTY25AUG24550CE")
out_all = os.environ.get("OUT_ALL", "nfo_full.csv")

kite = KiteConnect(api_key=api_key); kite.set_access_token(access)
print("🔍 Fetching NFO instruments…")
df = pd.DataFrame(kite.instruments("NFO"))
df.to_csv(out_all, index=False)
m = df[df["tradingsymbol"] == symbol]
print("✅ Match found!" if not m.empty else f"❌ No match for {symbol}")
PY