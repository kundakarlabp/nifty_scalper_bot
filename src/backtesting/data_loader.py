# src/backtesting/data_loader.py
import pandas as pd
import logging

logger = logging.getLogger(__name__)

def load_historical_data(filepath_or_source):
    """
    Loads historical OHLC data for backtesting from a CSV file.

    Args:
        filepath_or_source (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame with OHLCV data, indexed by datetime.
    """
    try:
        df = pd.read_csv(filepath_or_source, index_col='timestamp', parse_dates=True)
        df.sort_index(inplace=True)
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"Missing required columns: {required_columns}")
        logger.info(f"Loaded {len(df)} rows of historical data from {filepath_or_source}")
        return df
    except Exception as e:
        logger.error(f"Error loading historical data: {e}")
        return pd.DataFrame()
