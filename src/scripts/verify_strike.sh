#!/usr/bin/env bash
set -euo pipefail

# Optional .env
[[ -f .env ]] && source .env

: "${ZERODHA_API_KEY:?ZERODHA_API_KEY not set}"
: "${ZERODHA_ACCESS_TOKEN:?ZERODHA_ACCESS_TOKEN not set}"

TARGET="${1:-NIFTY25AUG24550CE}"
FULL_OUT="${FULL_OUT:-nfo_instruments_full.csv}"
WEEKLY_OUT="${WEEKLY_OUT:-nifty_weekly_options.csv}"

python3 - <<'PY'
import os, pandas as pd
from kiteconnect import KiteConnect

api_key = os.environ["ZERODHA_API_KEY"]
access  = os.environ["ZERODHA_ACCESS_TOKEN"]
target  = os.environ.get("TARGET", "NIFTY25AUG24550CE")
full_out = os.environ.get("FULL_OUT", "nfo_instruments_full.csv")
weekly_out = os.environ.get("WEEKLY_OUT", "nifty_weekly_options.csv")

kite = KiteConnect(api_key=api_key); kite.set_access_token(access)

print("🔍 Fetching ALL NFO instruments…")
rows = kite.instruments("NFO") or []
df = pd.DataFrame(rows)
df.to_csv(full_out, index=False)
print(f"✅ Saved full list → {full_out}")

nifty_opts = df[(df["name"]=="NIFTY") & (df["instrument_type"].isin(["CE","PE"]))]
nifty_opts.to_csv(weekly_out, index=False)
print(f"✅ Saved weekly options → {weekly_out}")

match = nifty_opts[nifty_opts["tradingsymbol"]==target]
print(f"✅ Symbol found: {target}" if not match.empty else f"❌ Symbol NOT found: {target}")
PY