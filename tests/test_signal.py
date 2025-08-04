"""
Manual test script to validate signal generation and order execution.
Runs in simulation mode — safe for testing in Codespaces.
"""

import pandas as pd
import logging
from datetime import datetime, timedelta

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Suppress noisy INFO logs from requests, telegram
for name in ['requests', 'urllib3', 'telegram']:
    logging.getLogger(name).setLevel(logging.WARNING)

from src.data_streaming.realtime_trader import RealTimeTrader


def create_test_candle(
    close: float,
    open_price: float = None,
    high: float = None,
    low: float = None,
    volume: int = 100000,
    symbol: str = "NIFTY"
) -> pd.DataFrame:
    """Create a synthetic 1-minute OHLC candle."""
    if open_price is None:
        open_price = close - (close * 0.001)
    if high is None:
        high = close + (close * 0.002)
    if low is None:
        low = close - (close * 0.002)

    df = pd.DataFrame([{
        "symbol": symbol,
        "open": open_price,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    }], index=[datetime.now() - timedelta(minutes=1)])  # Use datetime index

    # Ensure index is DatetimeIndex
    df.index = pd.DatetimeIndex(df.index)
    return df


def main():
    print("🧪 Starting Signal & Order Execution Test (Shadow Mode)")
    print("ℹ️  LIVE TRADING IS DISABLED — ALL ORDERS ARE SIMULATED")

    # Initialize the trader
    trader = RealTimeTrader()

    # Force shadow mode (double safe)
    trader.live_mode = False
    print(f"🛡️  Mode: {'LIVE' if trader.live_mode else 'SHADOW (SIMULATION)'}")

    # Start the trader (enables processing)
    trader.start()

    # Test Case 1: Bullish price
    print("\n📈 Testing Bullish Signal...")
    bullish_bar = create_test_candle(close=19500.5, open_price=19480.0)
    trader.process_bar(bullish_bar)

    # Test Case 2: Bearish price
    print("\n📉 Testing Bearish Signal...")
    bearish_bar = create_test_candle(close=19400.0, open_price=19420.0)
    trader.process_bar(bearish_bar)

    # Final status
    print("\n📊 Final Status:")
    status = trader.get_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    print("\n✅ Test complete. No real orders were placed.")


if __name__ == "__main__":
    main()
