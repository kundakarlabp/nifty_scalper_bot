import pandas as pd
import logging
from kiteconnect import KiteConnect
from datetime import datetime

logger = logging.getLogger(__name__)

def load_zerodha_historical_data(kite: KiteConnect, instrument_token: int, from_date: str, to_date: str, interval: str = "5minute") -> pd.DataFrame:
    """
    Loads historical data from Zerodha Kite Connect API.

    Args:
        kite (KiteConnect): Authenticated KiteConnect object
        instrument_token (int): Zerodha instrument token (e.g., 256265 for NIFTY)
        from_date (str): Start date in 'YYYY-MM-DD'
        to_date (str): End date in 'YYYY-MM-DD'
        interval (str): Interval like 'day', '5minute', etc.

    Returns:
        pd.DataFrame: OHLCV data indexed by timestamp
    """
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")

        data = kite.historical_data(instrument_token, from_dt, to_dt, interval)
        df = pd.DataFrame(data)

        if df.empty:
            logger.error("Zerodha returned empty historical data.")
            return pd.DataFrame()

        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df.drop(columns=['date'], inplace=True)
        df.sort_index(inplace=True)

        return df

    except Exception as e:
        logger.error(f"Error loading historical data from Zerodha: {e}")
        return pd.DataFrame()
