# config.py
import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables from .env (for local dev)
load_dotenv(dotenv_path=Path('.') / '.env')

class Config:
    # --- Zerodha Credentials ---
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "") # Provide defaults
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # --- Telegram Bot ---
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID: int = 0  # Fallback

    # --- Core Risk Settings ---
    MAX_RISK_PER_TRADE: float = float(os.getenv("MAX_RISK_PER_TRADE", 0.01))
    DAILY_MAX_LOSS: float = float(os.getenv("DAILY_MAX_LOSS", 0.05))
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # --- Trade Filters & Strategy Parameters ---
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))
    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    # --- Strategy Specific Parameters ---
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0)) # Check if this should be 0.8 or 8.0 based on strategy logic

    # --- Risk Management Parameters ---
    ACCOUNT_SIZE: float = float(os.getenv("ACCOUNT_SIZE", 100000.0))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", 0.05))

    # --- Instrument Settings ---
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", 75)) # Corrected default lot size for Nifty Options
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", 1))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", 5))

    # --- Logging & Files ---
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

    # --- Execution Control ---
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "False").lower() in ("true", "1", "yes")

    # --- Other Constants ---
    # FIXED: Corrected syntax error
    MIS: str = "MIS"
    DEFAULT_PRODUCT: str = MIS # This now correctly assigns the string "MIS"

# Example .env file content:
# ZERODHA_API_KEY=your_api_key_here
# ZERODHA_API_SECRET=your_api_secret_here
# KITE_ACCESS_TOKEN=your_request_token_here
# TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
# TELEGRAM_USER_ID=your_telegram_user_id_here
# ENABLE_LIVE_TRADING=True
# ACCOUNT_SIZE=500000.0
