# src/config.py
"""
Centralised configuration loader for the scalper bot.
Reads environment variables at import time and exposes them via the `Config` class.
"""

import os
from pathlib import Path

# Load .env file if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"🔐 Loaded environment from {env_path}")
    else:
        print("⚠️  .env file not found at:", env_path)
except ImportError:
    print("⚠️  python-dotenv not installed, skipping .env load.")
except Exception as exc:
    print(f"❌ Failed to load .env: {exc}")

if 'env_path' not in locals():
    env_path = Path('.env')


class Config:
    """Static configuration values pulled from environment variables."""

    # ——— ZERODHA ———
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")

    # ——— TELEGRAM ———
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_CHAT_ID: int = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
    except (ValueError, TypeError):
        print("⚠️  Invalid TELEGRAM_CHAT_ID, defaulting to 0")
        TELEGRAM_CHAT_ID = 0

    # ——— RISK/MONEY MGMT ———
    RISK_PER_TRADE: float = float(os.getenv("RISK_PER_TRADE", "0.01"))
    MAX_DRAWDOWN: float = float(os.getenv("MAX_DRAWDOWN", "0.05"))
    CONSECUTIVE_LOSS_LIMIT: int = int(os.getenv("CONSECUTIVE_LOSS_LIMIT", "3"))

    # ——— STRATEGY TUNING ———
    BASE_STOP_LOSS_POINTS: float = float(os.getenv("BASE_STOP_LOSS_POINTS", "20.0"))
    BASE_TARGET_POINTS: float = float(os.getenv("BASE_TARGET_POINTS", "40.0"))
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "8.0"))
    MIN_SIGNAL_SCORE: float = float(os.getenv("MIN_SIGNAL_SCORE", "7.0"))

    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    ATR_SL_MULTIPLIER: float = float(os.getenv("ATR_SL_MULTIPLIER", "1.5"))
    ATR_TP_MULTIPLIER: float = float(os.getenv("ATR_TP_MULTIPLIER", "3.0"))

    SL_CONFIDENCE_ADJ: float = float(os.getenv("SL_CONFIDENCE_ADJ", "0.2"))
    TP_CONFIDENCE_ADJ: float = float(os.getenv("TP_CONFIDENCE_ADJ", "0.3"))

    # ——— INSTRUMENTS ———
    NIFTY_LOT_SIZE: int = int(os.getenv("NIFTY_LOT_SIZE", "75"))
    MIN_LOTS: int = int(os.getenv("MIN_LOTS", "1"))
    MAX_LOTS: int = int(os.getenv("MAX_LOTS", "5"))

    TRADE_SYMBOL: str = os.getenv("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = os.getenv("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN: int = int(os.getenv("INSTRUMENT_TOKEN", "256265"))

    # Spot LTP symbol for kite.ltp
    SPOT_SYMBOL: str = os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50")
    OPTION_TYPE: str = os.getenv("OPTION_TYPE", "BOTH")            # CE/PE/BOTH
    STRIKE_SELECTION_TYPE: str = os.getenv("STRIKE_SELECTION_TYPE", "ATM")
    STRIKE_RANGE: int = int(os.getenv("STRIKE_RANGE", "3"))
    DATA_LOOKBACK_MINUTES: int = int(os.getenv("DATA_LOOKBACK_MINUTES", "30"))

    # Options strategy params
    OPTION_SL_PERCENT: float = float(os.getenv("OPTION_SL_PERCENT", "0.05"))
    OPTION_TP_PERCENT: float = float(os.getenv("OPTION_TP_PERCENT", "0.15"))
    OPTION_BREAKOUT_PCT: float = float(os.getenv("OPTION_BREAKOUT_PCT", "0.01"))
    OPTION_SPOT_TREND_PCT: float = float(os.getenv("OPTION_SPOT_TREND_PCT", "0.005"))

    # ——— EXECUTION DEFAULTS ———
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # ——— FEATURE TOGGLES ———
    ENABLE_LIVE_TRADING: bool = os.getenv("ENABLE_LIVE_TRADING", "false").lower() in ("true", "1", "yes")
    ENABLE_TELEGRAM: bool = os.getenv("ENABLE_TELEGRAM", "true").lower() in ("true", "1", "yes")

    # ——— HOUSEKEEPING / TEST ———
    ALLOW_OFFHOURS_TESTING: bool = os.getenv("ALLOW_OFFHOURS_TESTING", "false").lower() in ("true","1","yes")
    BALANCE_LOG_INTERVAL_MIN: int = int(os.getenv("BALANCE_LOG_INTERVAL_MIN", "30"))

    # ——— LOGGING ———
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")


if __name__ == "__main__":
    import pprint
    print("🔐 Final Config:")
    pprint.pprint({
        k: v for k, v in Config.__dict__.items()
        if not k.startswith("__") and not callable(v)
    })