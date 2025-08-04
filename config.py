"""
Centralised configuration loader for the scalper bot.
Reads environment variables at import time and exposes them via the `Config` class.
"""

import os
from pathlib import Path


# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path('.') / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
except ImportError:
    pass  # Silent fallback if python-dotenv isn't installed


class Config:
    """Static configuration values pulled from environment variables."""

    # ———————————————— ZERODHA CREDENTIALS ———————————————— #
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # ———————————————— TELEGRAM BOT ———————————————— #
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    
    # ✅ Use TELEGRAM_CHAT_ID (what your TelegramController expects)
    try:
        TELEGRAM_CHAT_ID: int = int(os.getenv("TELEGRAM_CHAT_ID", 0))
    except (ValueError, TypeError):
        TELEGRAM_CHAT_ID = 0

    # Optional: keep TELEGRAM_USER_ID for backward compatibility
    try:
        TELEGRAM_USER_ID: int = int(os.getenv("TELEGRAM_USER_ID", TELEGRAM_CHAT_ID))
    except (ValueError, TypeError):
        TELEGRAM_USER_ID = 0

    # ———————————————— RISK & MONEY MANAGEMENT ———————————————— #
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", 0.01))      # 1% per trade
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", 0.05))          # 5% daily max loss
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", 3))

    # ———————————————— STRATEGY TUNING ———————————————— #
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", 20.0))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", 40.0))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 8.0))
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", 7.0))

    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    # ATR-based adaptive SL/TP
    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", 1.5))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", 3.0))

    # Confidence-based adjustments
    SL_CONFIDENCE_ADJ: float = float(os.getenv("SL_CONFIDENCE_ADJ", 0.2))
    TP_CONFIDENCE_ADJ: float = float(os.getenv("TP_CONFIDENCE_ADJ", 0.3))

    # ———————————————— INSTRUMENT SETTINGS ———————————————— #
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", 50))
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", 1))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", 5))

    # Trade symbol and exchange
    TRADE_SYMBOL: str = os.getenv("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = os.getenv("TRADE_EXCHANGE", "NFO")

    # ———————————————— EXECUTION DEFAULTS ———————————————— #
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")       # MIS (intraday), CNC, NRML
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")  # MARKET, LIMIT, SL, SL-M
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # ———————————————— FEATURE TOGGLES ———————————————— #
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ENABLE_TELEGRAM: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")

    # ———————————————— LOGGING ———————————————— #
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")


# ———————————————— DEBUG: Print loaded config (optional) ———————————————— #
# Uncomment below to debug config loading (remove in production)
# if __name__ == "__main__":
#     import pprint
#     print("🔐 Loaded Config:")
#     pprint.pprint({k: v for k, v in Config.__dict__.items() if not k.startswith("__")})
