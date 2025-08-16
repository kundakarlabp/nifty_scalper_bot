# src/backtesting/backtest_runner.py
"""
Backtest runner for the Nifty Scalper Bot.

Features
- Loads data from either a local CSV or Zerodha (KiteConnect)
- Minimal, robust CLI with sensible defaults
- Validates columns and date range
- Runs BacktestEngine and writes results to logs/
- Prints a one-line summary at the end

Usage:
  python -m src.backtesting.backtest_runner \
    --from 2024-06-01 --to 2024-06-30 --interval 5minute --token 256265
  # or from CSV:
  python -m src.backtesting.backtest_runner --csv data/nifty_ohlc.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure repo root on path for "config" when invoked as a script
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Config & modules
from config import Config  # noqa: E402
from src.backtesting.data_loader import load_zerodha_historical_data  # noqa: E402
from src.backtesting.backtest_engine import BacktestEngine  # noqa: E402

# Optional: load .env if present so Config can read env-backed values
try:  # noqa: SIM105
    from dotenv import load_dotenv  # type: ignore
    for p in (REPO_ROOT / ".env", Path.cwd() / ".env"):
        if p.exists():
            load_dotenv(p)
            break
except Exception:
    pass


# -------------------------- logging -------------------------- #

LOG_DIR = REPO_ROOT / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "backtest.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("backtest_runner")


# -------------------------- helpers -------------------------- #

REQUIRED_COLS = {"open", "high", "low", "close"}

def _normalize_ohlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Case-insensitive rename for OHLC columns; returns a copy."""
    lower_map = {c.lower(): c for c in df.columns}
    rename = {lower_map[c]: c for c in REQUIRED_COLS if c in lower_map and lower_map[c] != c}
    if rename:
        df = df.rename(columns=rename)
    return df


def _load_from_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = _normalize_ohlc_columns(df)
    if not REQUIRED_COLS.issubset(df.columns):
        raise ValueError(f"CSV missing columns {sorted(REQUIRED_COLS)}; got {df.columns.tolist()}")
    # Promote/ensure a datetime index named 'date' for the engine logs
    if "date" in df.columns:
        try:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")
        except Exception:
            pass
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, errors="coerce")
    return df


def _load_from_zerodha(token: int, from_date: str, to_date: str, interval: str) -> pd.DataFrame:
    # Late import to avoid hard dependency if running CSV-only backtests
    from kiteconnect import KiteConnect  # type: ignore

    api_key = getattr(Config, "ZERODHA_API_KEY", None)
    access_token = getattr(Config, "KITE_ACCESS_TOKEN", None) or getattr(Config, "ZERODHA_ACCESS_TOKEN", None)
    if not api_key or not access_token:
        raise RuntimeError("ZERODHA_API_KEY and KITE_ACCESS_TOKEN (or ZERODHA_ACCESS_TOKEN) are required in Config/env.")

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    df = load_zerodha_historical_data(kite, token, from_date, to_date, interval)
    if df is None or df.empty:
        raise RuntimeError("No historical data returned from Zerodha.")
    return df


def _write_results(metrics: dict, out_dir: Path, tag: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"results_{tag}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info("📄 Results written to %s", path)


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest runner for Nifty Scalper Bot")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="Path to CSV with OHLC data")
    src.add_argument("--token", type=int, help="Kite instrument token (e.g., 256265 for NIFTY)")

    p.add_argument("--from", dest="from_date", help="From date YYYY-MM-DD (for Zerodha source)")
    p.add_argument("--to", dest="to_date", help="To date YYYY-MM-DD (for Zerodha source)")
    p.add_argument("--interval", default="5minute", help="Interval (minute, 3minute, 5minute, 15minute, day)")
    p.add_argument("--symbol", default="NIFTY", help="Symbol label for reports")
    p.add_argument("--tag", default=datetime.now().strftime("%Y%m%d_%H%M%S"), help="Tag appended to outputs")
    return p.parse_args(argv)


# -------------------------- main -------------------------- #

def run_backtest(
    *,
    csv_file_path: Optional[str] = None,
    instrument_token: Optional[int] = None,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    interval: str = "5minute",
    symbol: str = "NIFTY",
    tag: Optional[str] = None,
) -> dict:
    """Programmatic entrypoint (also used by CLI wrapper below)."""
    tag = tag or datetime.now().strftime("%Y%m%d_%H%M%S")

    # Load data
    if csv_file_path:
        logger.info("📂 Loading OHLC from CSV: %s", csv_file_path)
        df = _load_from_csv(csv_file_path)
    else:
        if not (instrument_token and from_date and to_date):
            raise ValueError("--token, --from and --to are required when not using --csv.")
        logger.info("☁️  Fetching OHLC from Zerodha: token=%s from=%s to=%s interval=%s",
                    instrument_token, from_date, to_date, interval)
        df = _load_from_zerodha(instrument_token, from_date, to_date, interval)

    # Validate
    if df.empty:
        raise RuntimeError("No data available after load.")
    if not REQUIRED_COLS.issubset(df.columns):
        raise RuntimeError(f"Data missing columns: needed {sorted(REQUIRED_COLS)}, got {df.columns.tolist()}")

    start, end = (df.index.min(), df.index.max())
    logger.info("📊 Data ready: %d rows, range: %s → %s", len(df), start, end)

    # Engine
    engine = BacktestEngine(df, symbol=symbol, log_file=str(LOG_DIR / f"backtest_trades_{tag}.csv"))

    # Run
    t0 = datetime.now()
    metrics = engine.run()
    dt = (datetime.now() - t0).total_seconds()
    logger.info("✅ Backtest complete in %.2fs | trades=%s | win=%.1f%% | net=%.2f | avgR=%.3f | maxDD(R)=%.3f",
                dt, metrics.get("trades", 0), metrics.get("win_rate", 0.0),
                metrics.get("net_pnl", 0.0), metrics.get("avg_R", 0.0),
                metrics.get("max_dd_R", 0.0))

    # Save JSON summary
    _write_results(metrics, LOG_DIR, tag)
    return metrics


def main(argv: Optional[list[str]] = None) -> None:
    args = _parse_args(argv)

    try:
        run_backtest(
            csv_file_path=args.csv,
            instrument_token=args.token,
            from_date=args.from_date,
            to_date=args.to_date,
            interval=args.interval,
            symbol=args.symbol,
            tag=args.tag,
        )
    except Exception as exc:
        logger.error("💥 Backtest failed: %s", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()