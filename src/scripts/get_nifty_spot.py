#!/usr/bin/env python3
"""
Print spot/last price for a symbol (default: NSE:NIFTY 50).

Usage:
  python -m src.scripts.get_nifty_spot.py
  python -m src.scripts.get_nifty_spot.py --symbol NSE:NIFTY\ 50
"""

from __future__ import annotations
import argparse
import json
import os
import sys

def _load_env():
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

def _kite():
    from kiteconnect import KiteConnect
    api_key = os.environ.get("ZERODHA_API_KEY")
    access_token = os.environ.get("ZERODHA_ACCESS_TOKEN")
    if not api_key or not access_token:
        print("❌ Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN.", file=sys.stderr)
        sys.exit(2)
    k = KiteConnect(api_key=api_key)
    k.set_access_token(access_token)
    return k

def main():
    _load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="NSE:NIFTY 50")
    ap.add_argument("--json", action="store_true", help="Print JSON row instead of plain text")
    args = ap.parse_args()

    k = _kite()
    q = k.quote(args.symbol)
    row = q.get(args.symbol) or {}
    last = float(row.get("last_price") or 0.0)
    if args.json:
        print(json.dumps({"symbol": args.symbol, "last_price": last}, separators=(",",":")))
    else:
        print(f"✅ {args.symbol} last_price: {last}")

if __name__ == "__main__":
    main()