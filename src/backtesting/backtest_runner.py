# src/backtesting/backtest_runner.py
"""
Main entry point for running backtests for the Nifty Scalper Bot.
Loads historical data, initializes the strategy and backtest engine, and executes the backtest.
"""

import os
import sys
import logging
from datetime import datetime
from typing import Optional
import pandas as pd

# Ensure correct path resolution for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import configuration
from config import Config

# Import KiteConnect
from kiteconnect import KiteConnect

# Import backtesting modules
from src.backtesting.data_loader import load_zerodha_historical_data
from src.backtesting.backtest_engine import BacktestEngine

# Import the strategy used by the live bot for consistency
# You'll need to adapt your BacktestEngine to accept and use this strategy
# from src.strategies.scalping_strategy import DynamicScalpingStrategy

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/backtest.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def run_backtest(
    instrument_token: int,
    from_date: str,
    to_date: str,
    interval: str = "5minute", # Or "minute" for 1-min data
    csv_file_path: Optional[str] = None # Optional: Load from local CSV if Kite fails/is slow
) -> None:
    """
    Runs a backtest for a given instrument and date range.

    Args:
        instrument_token (int): The Kite instrument token.
        from_date (str): Start date in 'YYYY-MM-DD' format.
        to_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval (e.g., 'minute', '5minute', ' hour').
        csv_file_path (str, optional): Path to a local CSV file containing historical data.
    """
    logger.info("🚀 Starting Backtest Runner...")
    logger.info(f"📋 Parameters: Token={instrument_token}, From={from_date}, To={to_date}, Interval={interval}")

    df: pd.DataFrame = pd.DataFrame() # Initialize empty DataFrame

    # --- 1. Load Data ---
    if csv_file_path and os.path.exists(csv_file_path):
        logger.info(f"📂 Loading data from local CSV: {csv_file_path}")
        try:
            df = pd.read_csv(csv_file_path, index_col='date', parse_dates=True)
            # Ensure column names match expected format (date, open, high, low, close, volume)
            # Adjust column names if your CSV uses different ones
            # df.rename(columns={'<ticker>': 'open', '<high>': 'high', ...}, inplace=True)
            logger.info(f"✅ Loaded {len(df)} rows from CSV.")
        except Exception as e:
            logger.error(f"❌ Error loading CSV data: {e}")
            return
    else:
        # Load from Zerodha Kite
        if not Config.ZERODHA_API_KEY or not Config.KITE_ACCESS_TOKEN:
            logger.critical("❌ Zerodha API credentials (ZERODHA_API_KEY, KITE_ACCESS_TOKEN) missing in config.")
            return

        kite = KiteConnect(api_key=Config.ZERODHA_API_KEY)
        try:
            kite.set_access_token(Config.KITE_ACCESS_TOKEN)
            logger.info("✅ Zerodha Kite Connect client initialized.")
        except Exception as e:
            logger.critical(f"❌ Failed to authenticate Kite client: {e}")
            return

        logger.info("📥 Fetching historical data from Zerodha Kite...")
        try:
            df = load_zerodha_historical_data(kite, instrument_token, from_date, to_date, interval)
            if df.empty:
                logger.error("❌ No historical data loaded from Zerodha. Exiting.")
                return
            logger.info(f"✅ Successfully loaded {len(df)} rows of historical data from Zerodha.")
        except Exception as e:
            logger.error(f"❌ Error fetching data from Zerodha: {e}")
            return

    # --- 2. Validate Data ---
    if df.empty:
        logger.error("❌ No data available for backtesting. Exiting.")
        return

    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"❌ Data missing required columns. Found: {df.columns.tolist()}")
        return

    logger.info(f"📊 Data Summary: {len(df)} rows, Date Range: {df.index.min()} to {df.index.max()}")

    # --- 3. Initialize Strategy (Conceptual) ---
    # To make the backtest truly representative of your live bot,
    # you would instantiate your live strategy here.
    # strategy = DynamicScalpingStrategy(
    #     base_stop_loss_points=Config.BASE_STOP_LOSS_POINTS,
    #     base_target_points=Config.BASE_TARGET_POINTS,
    #     confidence_threshold=Config.CONFIDENCE_THRESHOLD
    # )
    # Then pass this strategy to the BacktestEngine
    # engine = BacktestEngine(df, strategy=strategy)

    # --- 4. Initialize and Run Backtest Engine ---
    # Assuming your BacktestEngine can work with the raw DataFrame for now
    logger.info("⚙️ Initializing Backtest Engine...")
    engine = BacktestEngine(df) # Modify BacktestEngine's __init__ if passing strategy

    logger.info("🏁 Running Backtest...")
    start_time = datetime.now()
    try:
        engine.run()
        end_time = datetime.now()
        duration = end_time - start_time
        logger.info(f"✅ Backtest completed successfully in {duration.total_seconds():.2f} seconds.")
        
        # --- 5. Generate Report (Conceptual) ---
        # Add a method to your BacktestEngine to generate a summary report
        # report = engine.generate_report()
        # print(report)
        # logger.info("📄 Backtest Report Generated.")
        
    except Exception as e:
        logger.error(f"❌ Error during backtest execution: {e}", exc_info=True)
        return

    logger.info("🏁 Backtest Runner finished.")

if __name__ == "__main__":
    # --- Configuration ---
    # Example: Backtest Nifty 50 Index for June 2024 using 5-minute data
    INSTRUMENT_TOKEN = 256265  # Nifty 50 Index on NSE
    FROM_DATE = "2024-06-01"
    TO_DATE = "2024-06-30"
    INTERVAL = "5minute"
    # Optional: If you have saved data locally
    # CSV_PATH = "data/historical/nifty_50_2024_06_5min.csv"
    CSV_PATH = None

    # You could also make these command-line arguments
    # import argparse
    # parser = argparse.ArgumentParser(description='Run Backtest')
    # parser.add_argument('--token', type=int, default=256265, help='Instrument token')
    # parser.add_argument('--from', dest='from_date', default='2024-06-01', help='Start date (YYYY-MM-DD)')
    # ... add other args ...
    # args = parser.parse_args()

    run_backtest(
        instrument_token=INSTRUMENT_TOKEN,
        from_date=FROM_DATE,
        to_date=TO_DATE,
        interval=INTERVAL,
        csv_file_path=CSV_PATH
    )
