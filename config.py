# src/config.py
"""
Centralised configuration for the scalper bot.

- Loads .env if present
- Uses tolerant parsers so env like "8.0" works where an int is expected
- Exposes settings via the Config class (static attributes)
"""

from __future__ import annotations

import os
from pathlib import Path

# ------------------------------- .env loader ------------------------------- #

try:
    from dotenv import load_dotenv  # type: ignore
    # Resolve project root: src/config.py -> project/.env
    env_path = Path(__file__).resolve().parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"🔐 Loaded environment from {env_path}")
    else:
        print(f"⚠️  .env file not found at: {env_path}")
except Exception as exc:
    # Don't crash if dotenv isn't installed or other issues happen
    print(f"ℹ️ Skipping .env load ({exc})")

# ------------------------------- helpers ---------------------------------- #

def _getenv_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default

def _getenv_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None:
        return default
    return str(v).strip().lower() in ("true", "1", "yes", "y", "on")

def _getenv_float(key: str, default: float = 0.0) -> float:
    v = os.getenv(key)
    if v is None or v == "":
        return float(default)
    try:
        return float(v)
    except Exception:
        try:
            # Sometimes values come like "8," in certain locales; last resort cleanup
            return float(str(v).replace(",", "."))
        except Exception:
            return float(default)

def _getenv_int(key: str, default: int = 0) -> int:
    """
    Tolerant int parser:
    - Accepts "8" or "8.0" (casts via float first)
    - Falls back to default on any error
    """
    v = os.getenv(key)
    if v is None or v == "":
        return int(default)
    try:
        return int(float(v))
    except Exception:
        return int(default)

# ------------------------------- config ----------------------------------- #

class Config:
    """Static configuration values pulled from environment variables."""

    # ———————————————— ZERODHA CREDENTIALS ———————————————— #
    ZERODHA_API_KEY: str = _getenv_str("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = _getenv_str("ZERODHA_API_SECRET", "")
    KITE_ACCESS_TOKEN: str = _getenv_str("KITE_ACCESS_TOKEN", _getenv_str("ZERODHA_ACCESS_TOKEN", ""))  # fallback

    # ———————————————— TELEGRAM BOT ———————————————— #
    TELEGRAM_BOT_TOKEN: str = _getenv_str("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: int = _getenv_int("TELEGRAM_CHAT_ID", 0)

    # ———————————————— RISK & MONEY MANAGEMENT ———————————————— #
    RISK_PER_TRADE: float = _getenv_float("RISK_PER_TRADE", 0.01)
    MAX_DRAWDOWN: float = _getenv_float("MAX_DRAWDOWN", 0.05)
    CONSECUTIVE_LOSS_LIMIT: int = _getenv_int("CONSECUTIVE_LOSS_LIMIT", 3)

    # ———————————————— STRATEGY TUNING ———————————————— #
    BASE_STOP_LOSS_POINTS: float = _getenv_float("BASE_STOP_LOSS_POINTS", 20.0)
    BASE_TARGET_POINTS: float = _getenv_float("BASE_TARGET_POINTS", 40.0)
    # Accept "8" or "8.0" safely (was crashing earlier)
    CONFIDENCE_THRESHOLD: int = _getenv_int("CONFIDENCE_THRESHOLD", 8)
    MIN_SIGNAL_SCORE: int = _getenv_int("MIN_SIGNAL_SCORE", 7)

    TIME_FILTER_START: str = _getenv_str("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = _getenv_str("TIME_FILTER_END", "15:15")

    # ATR-based adaptive SL/TP
    ATR_SL_MULTIPLIER: float = _getenv_float("ATR_SL_MULTIPLIER", 1.5)
    ATR_TP_MULTIPLIER: float = _getenv_float("ATR_TP_MULTIPLIER", 3.0)

    # Confidence-based adjustments
    SL_CONFIDENCE_ADJ: float = _getenv_float("SL_CONFIDENCE_ADJ", 0.2)
    TP_CONFIDENCE_ADJ: float = _getenv_float("TP_CONFIDENCE_ADJ", 0.3)

    # ———————————————— INSTRUMENT SETTINGS ———————————————— #
    NIFTY_LOT_SIZE: int = _getenv_int("NIFTY_LOT_SIZE", 75)
    MIN_LOTS: int = _getenv_int("MIN_LOTS", 1)
    MAX_LOTS: int = _getenv_int("MAX_LOTS", 5)

    TRADE_SYMBOL: str = _getenv_str("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = _getenv_str("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN: int = _getenv_int("INSTRUMENT_TOKEN", 256265)  # verify for your instrument

    # Options / spot LTP symbol
    SPOT_SYMBOL: str = _getenv_str("SPOT_SYMBOL", "NSE:NIFTY 50")
    OPTION_TYPE: str = _getenv_str("OPTION_TYPE", "BOTH")  # CE, PE, BOTH
    STRIKE_SELECTION_TYPE: str = _getenv_str("STRIKE_SELECTION_TYPE", "ATM")  # ATM, ITM, OTM, OI_DELTA
    STRIKE_RANGE: int = _getenv_int("STRIKE_RANGE", 3)
    DATA_LOOKBACK_MINUTES: int = _getenv_int("DATA_LOOKBACK_MINUTES", 30)

    # Options scalping tuning
    OPTION_SL_PERCENT: float = _getenv_float("OPTION_SL_PERCENT", 0.05)
    OPTION_TP_PERCENT: float = _getenv_float("OPTION_TP_PERCENT", 0.15)
    OPTION_BREAKOUT_PCT: float = _getenv_float("OPTION_BREAKOUT_PCT", 0.01)
    OPTION_SPOT_TREND_PCT: float = _getenv_float("OPTION_SPOT_TREND_PCT", 0.005)

    # ———————————————— EXECUTION DEFAULTS ———————————————— #
    DEFAULT_PRODUCT: str = _getenv_str("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = _getenv_str("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = _getenv_str("DEFAULT_VALIDITY", "DAY")

    # Exit order preferences used by OrderExecutor
    DEFAULT_ORDER_TYPE_EXIT: str = _getenv_str("DEFAULT_ORDER_TYPE_EXIT", "LIMIT")  # exit legs
    PREFERRED_EXIT_MODE: str = _getenv_str("PREFERRED_EXIT_MODE", "AUTO")  # AUTO | GTT | REGULAR
    TICK_SIZE: float = _getenv_float("TICK_SIZE", 0.05)  # exchange tick size for NFO options
    TRAIL_COOLDOWN_SEC: float = _getenv_float("TRAIL_COOLDOWN_SEC", 12.0)

    # ———————————————— FEATURE TOGGLES ———————————————— #
    ENABLE_LIVE_TRADING: bool = _getenv_bool("ENABLE_LIVE_TRADING", False)
    ENABLE_TELEGRAM: bool = _getenv_bool("ENABLE_TELEGRAM", True)

    # ———————————————— LOGGING ———————————————— #
    LOG_FILE: str = _getenv_str("LOG_FILE", "logs/trades.csv")


# Debug print (optional): comment out in production
if __name__ == "__main__":
    import pprint
    cfg = {
        k: v
        for k, v in Config.__dict__.items()
        if not k.startswith("__") and not callable(v)
    }
    print("🔐 Final Config:")
    pprint.pprint(cfg)