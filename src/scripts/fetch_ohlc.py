#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fetch OHLC candles from Kite and save to CSV.

Features
- Reads API creds from env (.env supported)
- Accepts either instrument_token or tradingsymbol (e.g. NFO:NIFTY24AUG24600CE)
- Auto-chunks long ranges to respect Kite limits
- Optional OI column
- Robust CSV writing with path auto-create
- Retries with backoff on transient errors
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

import pandas as pd

# Optional: python-dotenv if installed (won't crash if missing)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from kiteconnect import KiteConnect


def _env(key: str, default: Optional[str] = None) -> str:
    v = os.environ.get(key, default)
    if v is None:
        raise RuntimeError(f"Missing environment variable: {key}")
    return v


def _parse_date(s: str) -> datetime:
    # Accept YYYY-MM-DD or full ISO 8601
    try:
        if len(s) == 10:
            return datetime.strptime(s, "%Y-%m-%d")
        return datetime.fromisoformat(s)
    except Exception:
        raise argparse.ArgumentTypeError(f"Invalid date: {s}")


def _chunks(start: datetime, end: datetime, interval: str) -> List[tuple[datetime, datetime]]:
    """
    Split request range into safe windows. Kite daily limits are generous,
    but minute endpoints work best with ≤60 days per call.
    """
    spans: List[tuple[datetime, datetime]] = []
    if interval.endswith("minute"):
        max_days = 60
    else:
        max_days = 200  # conservative for day/others

    cur = start
    while cur < end:
        nxt = min(cur + timedelta(days=max_days), end)
        spans.append((cur, nxt))
        cur = nxt
    return spans


def _historical(
    kite: KiteConnect,
    instrument_token: int,
    start: datetime,
    end: datetime,
    interval: str,
    include_oi: bool,
    tries: int = 3,
    backoff: float = 0.75,
) -> List[Dict[str, Any]]:
    last_exc = None
    for i in range(tries):
        try:
            return kite.historical_data(
                instrument_token=instrument_token,
                from_date=start,
                to_date=end,
                interval=interval,
                continuous=False,
                oi=include_oi,
            ) or []
        except Exception as exc:
            last_exc = exc
            if i == tries - 1:
                break
            sleep_s = backoff * (2 ** i)
            time.sleep(sleep_s)
    raise last_exc  # type: ignore


def _lookup_token(kite: KiteConnect, tradingsymbol: str) -> int:
    """
    Resolve "EXCHANGE:SYMBOL" → token using instruments cache.
    Examples: "NSE:NIFTY 50", "NFO:NIFTY24AUG24600CE"
    """
    if ":" not in tradingsymbol:
        raise ValueError("tradingsymbol must be like 'NSE:XXXX' or 'NFO:XXXX'")
    exch, sym = tradingsymbol.split(":", 1)
    rows = kite.instruments(exch.upper()) or []
    for r in rows:
        if str(r.get("tradingsymbol", "")).strip().upper() == sym.strip().upper():
            tok = r.get("instrument_token")
            if tok:
                return int(tok)
    raise ValueError(f"Could not resolve token for {tradingsymbol}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch OHLC from Kite → CSV")
    ap.add_argument("--token", type=int, help="Instrument token")
    ap.add_argument("--symbol", type=str, help="Exchange-qualified symbol, e.g. 'NSE:NIFTY 50'")
    ap.add_argument("--from", dest="from_date", type=_parse_date, required=True, help="From date (YYYY-MM-DD or ISO)")
    ap.add_argument("--to", dest="to_date", type=_parse_date, required=True, help="To date (YYYY-MM-DD or ISO)")
    ap.add_argument("--interval", default="5minute", help="minute|3minute|5minute|15minute|30minute|60minute|day")
    ap.add_argument("--oi", action="store_true", help="Include OI where available")
    ap.add_argument("--out", default="data/ohlc.csv", help="Output CSV path")
    ap.add_argument("--drop-partial", action="store_true", help="Drop last (possibly partial) candle")
    args = ap.parse_args()

    if not args.token and not args.symbol:
        ap.error("Provide either --token or --symbol")

    api_key = _env("ZERODHA_API_KEY")
    access_token = _env("ZERODHA_ACCESS_TOKEN")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    token = args.token or _lookup_token(kite, args.symbol)  # type: ignore

    spans = _chunks(args.from_date, args.to_date, args.interval)
    frames: List[pd.DataFrame] = []

    for i, (s, e) in enumerate(spans, 1):
        data = _historical(kite, token, s, e, args.interval, include_oi=args.oi)
        df = pd.DataFrame(data)
        if df.empty:
            continue
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        frames.append(df)
        # Light pacing between chunks
        time.sleep(0.25)

    if not frames:
        print("No data returned for given range.", file=sys.stderr)
        return 2

    out = pd.concat(frames).sort_index()
    if args.drop_partial and len(out) > 1:
        out = out.iloc[:-1]

    # Ensure directory
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out.to_csv(args.out)
    print(f"✅ Saved: {args.out}  rows={len(out)}  first={out.index[0]}  last={out.index[-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())