# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load .env file from project root
load_dotenv(dotenv_path=Path('.') / '.env')

class Config:
    # --- Zerodha Credentials ---
    ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY")
    ZERODHA_API_SECRET = os.getenv("ZERODHA_API_SECRET")
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN")

    # --- Telegram Bot ---
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
    # Ensure TELEGRAM_USER_ID is an integer; provide a default or handle potential conversion error
    try:
        TELEGRAM_USER_ID = int(os.getenv("TELEGRAM_USER_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID = 0 # Default if conversion fails

    # --- Core Risk Settings ---
    # Risk per single trade (e.g., 0.01 = 1% of account size)
    MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", 0.01))
    # Maximum allowable daily loss (e.g., 0.05 = 5% of account size)
    DAILY_MAX_LOSS = float(os.getenv("DAILY_MAX_LOSS", 0.05))
    # Stop trading after this many consecutive losses
    CONSECUTIVE_LOSS_LIMIT = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # --- Trade Filters & Strategy Parameters ---
    # Minimum signal score to consider a trade
    MIN_SIGNAL_SCORE = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))
    # Market start time (IST)
    TIME_FILTER_START = os.getenv("TIME_FILTER_START", "09:15")
    # Market end time (IST)
    TIME_FILTER_END = os.getenv("TIME_FILTER_END", "15:15")

    # --- Strategy Specific Parameters (used in realtime_trader.py and scalping_strategy.py) ---
    # Base stop loss points used by strategy or risk manager
    BASE_STOP_LOSS_POINTS = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    # Base target points used by strategy
    BASE_TARGET_POINTS = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    # Minimum confidence threshold for signal generation
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0))

    # --- Risk Management Parameters (used in position_sizing.py) ---
    # Total account size (capital)
    ACCOUNT_SIZE = float(os.getenv("ACCOUNT_SIZE", 100000.0)) # Example default value
    # Risk percentage per trade for PositionSizing
    RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01)) # 1% default
    # Maximum daily drawdown percentage for PositionSizing
    MAX_DRAWDOWN = float(os.getenv("MAX_DRAWDOWN", 0.05))     # 5% default

    # --- Instrument Settings (used in position_sizing.py) ---
    # Lot size for Nifty options
    NIFTY_LOT_SIZE = int(os.getenv("NIFTY_LOT_SIZE", 50)) # Default for Nifty Index Options
    # Minimum number of lots per trade
    MIN_LOTS = int(os.getenv("MIN_LOTS", 1))
    # Maximum number of lots per trade
    MAX_LOTS = int(os.getenv("MAX_LOTS", 5))

    # --- Paths ---
    LOG_FILE = os.getenv("LOG_FILE", "logs/trades.csv")
    MIS = "MIS"
    DEFAULT_PRODUCT =MIS

