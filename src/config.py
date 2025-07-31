"""
Centralised configuration loader for the scalper bot.  Reads environment
variables at import time and exposes them via the ``Config`` class.  All
credentials and tuning parameters should be defined in a ``.env`` file at
the project root (see ``.env.example``) or provided via the runtime
environment.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file if present
load_dotenv(dotenv_path=Path('.') / '.env')


class Config:
    """
    Static configuration values pulled from environment variables.  Default
    values are provided for most fields to allow local development without
    a .env file.  See ``.env.example`` for a full list of supported keys.
    """
    # Zerodha credentials
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # Telegram bot
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID = 0

    # Risk and money management
    ACCOUNT_SIZE: float = float(os.getenv("ACCOUNT_SIZE", 100_000.0))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))  # 1% of capital per trade
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", 0.05))      # 5% daily drawdown limit
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # Strategy tuning
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0))
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))  # external threshold (0–10 scale)
    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    # Instrument settings
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", 50))
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", 1))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", 5))

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

    # Execution defaults
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # Feature toggles
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ENABLE_TELEGRAM: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")