#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Print LTP/quote for one or more exchange-qualified symbols.

Examples
  python get_nifty_spot.py --symbol NSE:NIFTY 50
  python get_nifty_spot.py --symbol NSE:NIFTY BANK --ltp
  python get_nifty_spot.py --symbol NSE:NIFTY 50 --symbol NFO:NIFTY24AUG24600CE
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import List, Dict, Any

# Optional dotenv
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except Exception:
    pass

from kiteconnect import KiteConnect


def _env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"Missing env var: {key}")
    return v


def _ltp_batch(kite: KiteConnect, symbols: List[str]) -> Dict[str, Any]:
    # retry a bit for transient issues
    last_exc = None
    for i in range(3):
        try:
            return kite.ltp(symbols) or {}
        except Exception as exc:
            last_exc = exc
            time.sleep(0.5 * (2 ** i))
    raise last_exc  # type: ignore


def main() -> int:
    ap = argparse.ArgumentParser(description="Get quotes/LTP for symbols")
    ap.add_argument("--symbol", action="append", required=True, help="Exchange:Tradingsymbol (repeatable)")
    ap.add_argument("--ltp", action="store_true", help="Print only last_price values")
    args = ap.parse_args()

    kite = KiteConnect(api_key=_env("ZERODHA_API_KEY"))
    kite.set_access_token(_env("ZERODHA_ACCESS_TOKEN"))

    try:
        if args.ltp:
            out = _ltp_batch(kite, args.symbol)
            for k in args.symbol:
                lp = (out.get(k) or {}).get("last_price")
                print(f"{k}: {lp}")
            return 0

        # full quote (slower single-call)
        quotes = kite.quote(args.symbol) or {}
        for k in args.symbol:
            q = quotes.get(k) or {}
            lp = q.get("last_price")
            oi = q.get("oi")
            o = q.get("ohlc", {})
            print(f"{k}: LTP={lp}  OI={oi}  O={o.get('open')} H={o.get('high')} L={o.get('low')} C={o.get('close')}")
        return 0
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())