#!/usr/bin/env python3
"""
AMO Smoke Test: Validates Kite auth, order placement, status polling, and cancellation.
Safe off-hours via AMO + LIMIT far from LTP, then immediate cancel.

Usage:
  # Equity (INFY)
  python tools/amo_smoke.py

  # NFO option (provide a real tradingsymbol, e.g. NIFTY25SEP24600CE)
  python tools/amo_smoke.py --nfo --ts NIFTY25SEP24600CE --qty 50
"""
from __future__ import annotations
import os
import sys
import argparse
import logging
from kiteconnect import KiteConnect

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
log = logging.getLogger("amo_smoke")


def _require_env(key: str) -> str:
    val = os.getenv(key)
    if not val:
        log.error("Missing env: %s", key)
        sys.exit(1)
    return val


def _kite() -> KiteConnect:
    api = _require_env("KITE_API_KEY")
    access = _require_env("KITE_ACCESS_TOKEN")
    k = KiteConnect(api_key=api)
    k.set_access_token(access)
    return k


def _pct_price(ltp: float, pct: float) -> float:
    # round to 2 decimals; for NFO options this snaps fine to 0.05 tick on most strikes
    p = round(ltp * pct, 2)
    # keep within basic circuit sanity; broker will validate again anyway
    return max(p, 0.05)


def run_equity(kite: KiteConnect) -> None:
    symbol = "NSE:INFY"
    q = kite.quote(symbol)[symbol]
    ltp = float(q["last_price"])
    price = _pct_price(ltp, 0.90)  # 10% below — very unlikely to fill
    log.info("Equity: %s LTP=%.2f AMO LIMIT=%.2f", symbol, ltp, price)

    oid = kite.place_order(
        variety="amo",
        exchange="NSE",
        tradingsymbol="INFY",
        transaction_type="BUY",
        quantity=1,
        product="CNC",
        order_type="LIMIT",
        price=price,
        validity="DAY",
    )
    log.info("✅ AMO placed: %s", oid)
    st = kite.order_history(oid)[-1].get("status")
    log.info("📊 Status: %s", st)
    kite.cancel_order(variety="amo", order_id=oid)
    log.info("🧹 Cancelled AMO order")


def run_nfo(kite: KiteConnect, ts: str, qty: int) -> None:
    """
    NFO AMO test for a given option tradingsymbol, e.g. NIFTY25SEP24600CE
    - AMO for index options MUST be LIMIT (not MARKET).
    - Use BUY test (no margin impact), minimal risk, immediate cancel.
    """
    full = f"NFO:{ts}"
    q = kite.quote(full)[full]
    ltp = float(q["last_price"])
    # 20% below LTP; adjust if you get 'price outside range' — try 0.95 then.
    price = _pct_price(ltp, 0.80)
    log.info("NFO: %s LTP=%.2f AMO LIMIT=%.2f QTY=%d", full, ltp, price, qty)

    oid = kite.place_order(
        variety="amo",
        exchange="NFO",
        tradingsymbol=ts,
        transaction_type="BUY",      # BUY to avoid margin complexity for smoke test
        quantity=qty,
        product="NRML",              # NRML for overnight eligibility (AMO-safe)
        order_type="LIMIT",
        price=price,
        validity="DAY",
    )
    log.info("✅ AMO placed: %s", oid)
    st = kite.order_history(oid)[-1].get("status")
    log.info("📊 Status: %s", st)
    kite.cancel_order(variety="amo", order_id=oid)
    log.info("🧹 Cancelled AMO order")


def main() -> int:
    ap = argparse.ArgumentParser(description="Kite AMO smoke test")
    ap.add_argument("--nfo", action="store_true", help="Run NFO option test")
    ap.add_argument("--ts", help="NFO tradingsymbol (e.g. NIFTY25SEP24600CE)")
    ap.add_argument("--qty", type=int, default=int(os.getenv("LOT_SIZE_DEFAULT", 50)),
                    help="Quantity (lot size for the option). Default from LOT_SIZE_DEFAULT or 50.")
    args = ap.parse_args()

    kite = _kite()
    try:
        if args.nfo:
            if not args.ts:
                log.error("Provide --ts for NFO test (e.g. --ts NIFTY25SEP24600CE)")
                return 2
            run_nfo(kite, args.ts, args.qty)
        else:
            run_equity(kite)
        log.info("✨ AMO smoke test PASSED")
        return 0
    except Exception as e:
        log.exception("💥 AMO smoke test FAILED: %s", e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
