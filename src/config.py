"""
Centralised configuration loader for the scalper bot.
Reads environment variables at import time and exposes them via the ``Config`` class.
All credentials and tuning parameters should be defined in a ``.env`` file at
the project root (see ``.env.example``) or provided via the runtime environment.
"""

import os
from pathlib import Path

# Attempt to import dotenv loader
try:
    from dotenv import load_dotenv  # type: ignore
except ImportError:
    def load_dotenv(*args, **kwargs) -> None:
        return None

# Load .env file if present
load_dotenv(dotenv_path=Path('.') / '.env')


class Config:
    """
    Static configuration values pulled from environment variables.
    Default values are provided for most fields to allow local development
    without a .env file.
    """

    # Zerodha credentials
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # Telegram bot credentials
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID = 0

    # Enable inline buttons and interactive controls
    TELEGRAM_USE_INLINE_BUTTONS: bool = os.getenv("TELEGRAM_USE_INLINE_BUTTONS", "true").lower() in ("true", "1", "yes")

    # Trading session timing filter (HH:MM format)
    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    # Risk and money management
    ACCOUNT_SIZE: float = float(os.getenv("ACCOUNT_SIZE", 100_000.0))
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", 0.05))
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # Strategy tuning
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0))
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))

    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", 1.5))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", 3.0))

    SL_CONFIDENCE_ADJ: float = float(os.getenv("SL_CONFIDENCE_ADJ", 0.2))
    TP_CONFIDENCE_ADJ: float = float(os.getenv("TP_CONFIDENCE_ADJ", 0.3))

    # Instrument settings
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", 50))
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", 1))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", 5))

    # Execution and order placement
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # Logging
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

    # Feature toggles
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ENABLE_TELEGRAM: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")

    # Daily reset time (optional)
    DAILY_RESET_TIME: str = os.getenv("DAILY_RESET_TIME", "15:20")

    # Options filter (for advanced use)
    OPTION_VOLUME_THRESHOLD: int = int(os.getenv("OPTION_VOLUME_THRESHOLD", 10000))
    OI_CHANGE_THRESHOLD: float = float(os.getenv("OI_CHANGE_THRESHOLD", 5.0))  # %

