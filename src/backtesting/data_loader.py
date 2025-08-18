# src/backtesting/data_loader.py
import logging
from datetime import datetime
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def load_zerodha_historical_data(
    kite: Any,
    instrument_token: int,
    from_date: str,
    to_date: str,
    interval: str = "5minute",
) -> pd.DataFrame:
    """Load historical OHLCV from Zerodha KiteConnect API (import-safe)."""
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")
        logger.info("Fetch Zerodha data: token=%s interval=%s %s→%s", instrument_token, interval, from_date, to_date)
        data = kite.historical_data(instrument_token, from_dt, to_dt, interval)

        if not data:
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        df["datetime"] = pd.to_datetime(df["datetime"], utc=False).dt.tz_localize(None)
        df.set_index("datetime", inplace=True)
        keep = [c for c in ("open", "high", "low", "close", "volume") if c in df.columns]
        return df[keep].sort_index()
    except Exception as e:
        logger.error("Error loading historical data: %s", e, exc_info=True)
        return pd.DataFrame()
