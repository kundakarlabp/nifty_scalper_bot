#!/usr/bin/env python3
"""
Fetch OHLC candles from Kite and save to CSV.

Examples:
  python -m src.scripts.fetch_ohlc --token 256265 --from 2024-07-01 --to 2024-07-12 --interval 5minute
  python -m src.scripts.fetch_ohlc --symbol NSE:NIFTY\ 50 --from 2024-07-01 --to 2024-07-12

Env:
  ZERODHA_API_KEY, ZERODHA_ACCESS_TOKEN  (loaded from .env if present)
"""

from __future__ import annotations
import argparse
import os
import sys
from datetime import datetime
import pandas as pd

def _load_env():
    # load .env if present (without external deps)
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

def _parse_date(d: str) -> datetime:
    return datetime.strptime(d, "%Y-%m-%d")

def main():
    _load_env()

    p = argparse.ArgumentParser(description="Download OHLC to CSV")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--token", type=int, help="instrument_token")
    g.add_argument("--symbol", type=str, help="exchange:tradingsymbol (e.g. NSE:NIFTY 50)")
    p.add_argument("--from", dest="date_from", required=True, help="YYYY-MM-DD")
    p.add_argument("--to", dest="date_to", required=True, help="YYYY-MM-DD")
    p.add_argument("--interval", default="5minute",
                   choices=["minute","3minute","5minute","10minute","15minute","30minute","60minute","day"])
    p.add_argument("--out", default="data/ohlc.csv")
    args = p.parse_args()

    k = _kite()

    # Resolve token from symbol if needed
    token = args.token
    if not token:
        q = k.ltp([args.symbol])
        lp = q.get(args.symbol)
        if not lp:
            print(f"❌ LTP lookup failed for {args.symbol}", file=sys.stderr)
            sys.exit(3)
        token = lp.get("instrument_token") or lp.get("instrument_token", None)
        # Some Kite versions don't return token in ltp; fallback via instruments
        if not token:
            exch, ts = args.symbol.split(":", 1)
            cats = k.instruments(exch)
            for row in cats or []:
                if row.get("tradingsymbol") == ts:
                    token = row.get("instrument_token")
                    break
        if not token:
            print("❌ Could not resolve instrument_token from symbol.", file=sys.stderr)
            sys.exit(4)

    start, end = _parse_date(args.date_from), _parse_date(args.date_to)
    candles = k.historical_data(token, start, end, args.interval, oi=False) or []
    if not candles:
        print("⚠️ No candles returned.", file=sys.stderr)

    df = pd.DataFrame(candles)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out)
    print(f"✅ Saved {len(df):,} rows → {args.out}")

if __name__ == "__main__":
    main()