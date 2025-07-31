import os
from kiteconnect import KiteConnect
from src.backtesting.data_loader import load_zerodha_historical_data
from src.backtesting.backtest_engine import BacktestEngine
import logging

logging.basicConfig(level=logging.INFO)

def main():
    api_key = os.getenv("ZERODHA_API_KEY")
    access_token = os.getenv("ZERODHA_ACCESS_TOKEN")
    instrument_token = 256265  # NIFTY 50 spot index token

    if not api_key or not access_token:
        print("❌ Missing ZERODHA_API_KEY or ZERODHA_ACCESS_TOKEN in environment.")
        return

    kite = KiteConnect(api_key=api_key)
    kite.set_access_token(access_token)

    from_date = "2024-06-01"
    to_date = "2024-06-30"
    interval = "5minute"

    df = load_zerodha_historical_data(kite, instrument_token, from_date, to_date, interval)

    if df.empty:
        print("❌ No historical data loaded. Exiting.")
        return

    engine = BacktestEngine(df)
    engine.run()

if __name__ == "__main__":
    main()
