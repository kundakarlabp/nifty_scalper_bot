"""Main entry point for running backtests for the Nifty Scalper Bot."""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd

from src.config import Config
from src.backtesting.data_loader import load_zerodha_historical_data
from src.backtesting.backtest_engine import BacktestEngine

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

def run_backtest(
    api_key: str,
    access_token: str,
    instrument_token: int,
    start_date: str,
    end_date: str,
    interval: str = "5minute",
) -> pd.DataFrame:
    """Fetch data via KiteConnect and run backtest. Requires kiteconnect installed."""
    try:
        from kiteconnect import KiteConnect  # local import to avoid import-time crash
    except Exception as e:
        raise ImportError("kiteconnect is not installed. pip install kiteconnect") from e

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    df = load_zerodha_historical_data(kite, instrument_token, start_date, end_date, interval)
    if df.empty:
        logger.error("No data returned; aborting backtest.")
        return df

    engine = BacktestEngine(df, initial_capital=100000.0)
    result = engine.run()
    return result

if __name__ == "__main__":
    # Example / smoke test using Config; adjust as needed.
    result = run_backtest(
        api_key=Config.ZERODHA_API_KEY,
        access_token=Config.ZERODHA_ACCESS_TOKEN,
        instrument_token=256265,  # NIFTY 50 index token; replace as required
        start_date="2024-01-01",
        end_date="2024-01-10",
        interval="5minute",
    )
    print(result.tail())
