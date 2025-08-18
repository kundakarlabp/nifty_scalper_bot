# src/backtesting/sample_data_generator.py
"""
Synthetic 1-minute OHLCV generator (spot-like).

Usage:
  python -m src.backtesting.sample_data_generator --days 5 --out sample.csv
  python -m src.backtesting.sample_data_generator --from 2024-01-01 --to 2024-01-05 --out sample.csv
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import numpy as np
import pandas as pd


def _market_minutes(d0: datetime, d1: datetime):
    cur = d0
    while cur <= d1:
        if cur.weekday() < 5:
            start = cur.replace(hour=9, minute=15, second=0, microsecond=0)
            end = cur.replace(hour=15, minute=30, second=0, microsecond=0)
            t = start
            while t <= end:
                yield t
                t += timedelta(minutes=1)
        cur += timedelta(days=1)


def make_synthetic_ohlcv(date_from: str, date_to: str) -> pd.DataFrame:
    d0 = datetime.strptime(date_from, "%Y-%m-%d")
    d1 = datetime.strptime(date_to, "%Y-%m-%d")

    times = list(_market_minutes(d0, d1))
    n = len(times)
    if n == 0:
        raise ValueError("No market minutes in range")

    # Random walk with drift, inject regimes
    rng = np.random.default_rng(42)
    drift = 0.01
    noise = rng.normal(scale=0.6, size=n)
    regime = rng.choice([0.8, 1.2], size=n, p=[0.6, 0.4])
    price = 22500 + np.cumsum(drift * regime + noise).astype(float)

    # Build OHLCV
    close = price
    open_ = np.concatenate([[price[0]], price[:-1]])
    high = np.maximum(open_, close) + rng.random(n) * 0.8
    low = np.minimum(open_, close) - rng.random(n) * 0.8
    vol = rng.integers(low=1000, high=5000, size=n)

    df = pd.DataFrame(
        {"datetime": times, "open": open_, "high": high, "low": low, "close": close, "volume": vol}
    ).set_index("datetime")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--from", dest="date_from", default=None)
    ap.add_argument("--to", dest="date_to", default=None)
    ap.add_argument("--out", default="sample.csv")
    args = ap.parse_args()

    if args.days is not None:
        from_date = (datetime.now() - timedelta(days=args.days)).strftime("%Y-%m-%d")
        to_date = datetime.now().strftime("%Y-%m-%d")
    else:
        if not (args.date_from and args.date_to):
            raise SystemExit("Provide --days or both --from and --to.")
        from_date, to_date = args.date_from, args.date_to

    df = make_synthetic_ohlcv(from_date, to_date)
    df.to_csv(args.out)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
