import pandas as pd
import logging
from typing import Any
from datetime import datetime

logger = logging.getLogger(__name__)

def load_zerodha_historical_data(
    kite: Any,
    instrument_token: int,
    from_date: str,
    to_date: str,
    interval: str = "5minute",
) -> pd.DataFrame:
    """Load historical OHLCV data from Zerodha KiteConnect API.

    This keeps type-hints generic so the module can be imported without
    requiring the broker SDK at import time.
    """
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")

        logger.info("📥 Fetching Zerodha data: %s | %s | %s to %s", instrument_token, interval, from_date, to_date)
        data = kite.historical_data(instrument_token, from_dt, to_dt, interval)

        if not data:
            logger.warning("No historical data returned for token %s", instrument_token)
            return pd.DataFrame()

        df = pd.DataFrame(data)
        if "date" in df.columns:
            df.rename(columns={"date": "datetime"}, inplace=True)
        if "datetime" not in df.columns:
            logger.error("Historical data missing 'datetime' column")
            return pd.DataFrame()

        # Normalize timestamps
        if pd.api.types.is_datetime64_any_dtype(df["datetime"]):
            df["datetime"] = df["datetime"].dt.tz_localize(None)

        df.set_index("datetime", inplace=True)
        keep = [c for c in ("open","high","low","close","volume") if c in df.columns]
        return df[keep]
    except Exception as e:
        logger.error("Error loading historical data: %s", e, exc_info=True)
        return pd.DataFrame()
