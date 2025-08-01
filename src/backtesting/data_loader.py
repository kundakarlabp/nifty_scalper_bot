import pandas as pd
import logging
from kiteconnect import KiteConnect
from datetime import datetime

logger = logging.getLogger(__name__)

def load_zerodha_historical_data(
    kite: KiteConnect,
    instrument_token: int,
    from_date: str,
    to_date: str,
    interval: str = "5minute"
) -> pd.DataFrame:
    """
    Loads historical OHLCV data from Zerodha Kite Connect API.

    Args:
        kite (KiteConnect): Authenticated KiteConnect client.
        instrument_token (int): Zerodha instrument token (e.g., 256265 for NIFTY).
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Interval for data (e.g., 'day', '5minute', 'minute').

    Returns:
        pd.DataFrame: DataFrame with timestamp index and columns: open, high, low, close, volume.
    """
    try:
        from_dt = datetime.strptime(from_date, "%Y-%m-%d")
        to_dt = datetime.strptime(to_date, "%Y-%m-%d")

        logger.info(f"📥 Fetching Zerodha data: {instrument_token} | {interval} | {from_date} to {to_date}")
        data = kite.historical_data(instrument_token, from_dt, to_dt, interval)

        if not data:
            logger.warning("⚠️ Zerodha returned empty data.")
            return pd.DataFrame()

        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df.drop(columns=['date'], inplace=True, errors='ignore')
        df.sort_index(inplace=True)

        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.warning(f"⚠️ Missing columns in data: {missing_cols}")

        return df

    except Exception as e:
        logger.error(f"❌ Exception in loading Zerodha historical data: {e}", exc_info=True)
        return pd.DataFrame()
