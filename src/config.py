# src/config.py
"""
Centralised configuration loader for the scalper bot.
Reads environment variables at import time and exposes them via the `Config` class.
"""

import os
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"Loaded environment from {env_path}")
    else:
        print("No .env found at:", env_path)
except Exception as exc:
    print(f"dotenv not used: {exc}")


def _as_bool(val: str, default: bool = False) -> bool:
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "y", "on")


def _as_int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _as_float_env(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class Config:
    """Static configuration values pulled from environment variables."""

    # ─────────── Zerodha ─────────── #
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "")
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "")
    # Access token var name unified across project:
    KITE_ACCESS_TOKEN: str = os.getenv("KITE_ACCESS_TOKEN", "")
    # Back-compat (if someone sets ZERODHA_ACCESS_TOKEN)
    if not KITE_ACCESS_TOKEN and os.getenv("ZERODHA_ACCESS_TOKEN"):
        KITE_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN", "")

    # ─────────── Telegram ─────────── #
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    try:
        TELEGRAM_CHAT_ID: int = int(os.getenv("TELEGRAM_CHAT_ID", "0"))
    except Exception:
        TELEGRAM_CHAT_ID = 0

    # ─────────── Risk & Money Mgmt ─────────── #
    RISK_PER_TRADE: float = _as_float_env("RISK_PER_TRADE", 0.01)
    MAX_DRAWDOWN: float = _as_float_env("MAX_DRAWDOWN", 0.05)
    CONSECUTIVE_LOSS_LIMIT: int = _as_int_env("CONSECUTIVE_LOSS_LIMIT", 3)

    # Daily circuit breaker (amount OR pct). If both provided, stricter is used.
    DAILY_MAX_LOSS_PCT: float = _as_float_env("DAILY_MAX_LOSS_PCT", 0.05)   # legacy
    DAILY_MAX_LOSS_AMOUNT: float = _as_float_env("DAILY_MAX_LOSS_AMOUNT", 0.0)
    HALT_ON_DRAWDOWN: bool = _as_bool(os.getenv("HALT_ON_DRAWDOWN", "true"))

    # New: realtime_trader uses MAX_DAILY_DRAWDOWN_PCT (equivalent of DAILY_MAX_LOSS_PCT)
    MAX_DAILY_DRAWDOWN_PCT: float = _as_float_env("MAX_DAILY_DRAWDOWN_PCT", DAILY_MAX_LOSS_PCT)
    CIRCUIT_RELEASE_PCT: float = _as_float_env("CIRCUIT_RELEASE_PCT", 0.015)

    # ─────────── Strategy Tuning ─────────── #
    BASE_STOP_LOSS_POINTS: float = _as_float_env("BASE_STOP_LOSS_POINTS", 20.0)
    BASE_TARGET_POINTS: float = _as_float_env("BASE_TARGET_POINTS", 40.0)
    CONFIDENCE_THRESHOLD: float = _as_float_env("CONFIDENCE_THRESHOLD", 8.0)
    MIN_SIGNAL_SCORE: float = _as_float_env("MIN_SIGNAL_SCORE", 7.0)

    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")
    SKIP_FIRST_MIN: int = _as_int_env("SKIP_FIRST_MIN", 10)  # warmup ignore

    # ATR-based adaptive SL/TP
    ATR_SL_MULTIPLIER: float = _as_float_env("ATR_SL_MULTIPLIER", 1.5)
    ATR_TP_MULTIPLIER: float = _as_float_env("ATR_TP_MULTIPLIER", 3.0)
    ATR_PERIOD: int = _as_int_env("ATR_PERIOD", 14)
    ATR_LOOKBACK_MIN: int = _as_int_env("ATR_LOOKBACK_MIN", 30)

    # Confidence-based adjustments
    SL_CONFIDENCE_ADJ: float = _as_float_env("SL_CONFIDENCE_ADJ", 0.2)
    TP_CONFIDENCE_ADJ: float = _as_float_env("TP_CONFIDENCE_ADJ", 0.3)

    # ─────────── Instrument / Trading ─────────── #
    NIFTY_LOT_SIZE: int = _as_int_env("NIFTY_LOT_SIZE", 75)
    MIN_LOTS: int = _as_int_env("MIN_LOTS", 1)
    MAX_LOTS: int = _as_int_env("MAX_LOTS", 5)

    TRADE_SYMBOL: str = os.getenv("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = os.getenv("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN: int = _as_int_env("INSTRUMENT_TOKEN", 256265)

    # Options specifics
    SPOT_SYMBOL: str = os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50")
    OPTION_TYPE: str = os.getenv("OPTION_TYPE", "BOTH")  # CE, PE, BOTH
    STRIKE_SELECTION_TYPE: str = os.getenv("STRIKE_SELECTION_TYPE", "ATM")
    STRIKE_RANGE: int = _as_int_env("STRIKE_RANGE", 3)
    DATA_LOOKBACK_MINUTES: int = _as_int_env("DATA_LOOKBACK_MINUTES", 35)

    OPTION_SL_PERCENT: float = _as_float_env("OPTION_SL_PERCENT", 0.05)
    OPTION_TP_PERCENT: float = _as_float_env("OPTION_TP_PERCENT", 0.15)
    OPTION_BREAKOUT_PCT: float = _as_float_env("OPTION_BREAKOUT_PCT", 0.01)
    OPTION_SPOT_TREND_PCT: float = _as_float_env("OPTION_SPOT_TREND_PCT", 0.005)

    # Quotes / microstructure
    # Existing guard (unused by new trader) kept for compatibility:
    SPREAD_GUARD_MAX_PCT: float = _as_float_env("SPREAD_GUARD_MAX_PCT", 0.7)
    # New spread guard knobs used by realtime_trader:
    SPREAD_GUARD_MODE: str = os.getenv("SPREAD_GUARD_MODE", "LTP_MID")  # LTP_MID | RANGE
    SPREAD_GUARD_BA_MAX: float = _as_float_env("SPREAD_GUARD_BA_MAX", 0.012)       # 1.2% of mid
    SPREAD_GUARD_LTPMID_MAX: float = _as_float_env("SPREAD_GUARD_LTPMID_MAX", 0.015)  # 1.5% of mid
    SPREAD_GUARD_PCT: float = _as_float_env("SPREAD_GUARD_PCT", 0.02)  # RANGE proxy (2%)

    # Concurrency policy
    MAX_CONCURRENT_POSITIONS: int = _as_int_env("MAX_CONCURRENT_POSITIONS", 1)
    # New key used directly by realtime_trader (falls back to the above):
    MAX_CONCURRENT_TRADES: int = _as_int_env("MAX_CONCURRENT_TRADES", MAX_CONCURRENT_POSITIONS)
    WARMUP_BARS: int = _as_int_env("WARMUP_BARS", 60)

    # Slippage & fees
    SLIPPAGE_BPS: float = _as_float_env("SLIPPAGE_BPS", 4.0)        # 4 bps = 0.04% per side
    FEES_PCT_PER_SIDE: float = _as_float_env("FEES_PCT_PER_SIDE", 0.03)  # legacy (unused by new trader)
    FEES_PER_LOT: float = _as_float_env("FEES_PER_LOT", 25.0)       # ₹ per lot round trip, used by trader

    # Scheduling / rate limit
    JITTER_SECONDS_MAX: float = _as_float_env("JITTER_SECONDS_MAX", 2.0)
    BALANCE_LOG_INTERVAL_MIN: int = _as_int_env("BALANCE_LOG_INTERVAL_MIN", 30)
    WORKER_INTERVAL_SEC: int = _as_int_env("WORKER_INTERVAL_SEC", 10)

    # Order-exit preferences
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")
    PREFERRED_EXIT_MODE: str = os.getenv("PREFERRED_EXIT_MODE", "AUTO")  # AUTO | GTT | REGULAR
    TICK_SIZE: float = _as_float_env("TICK_SIZE", 0.05)
    TRAIL_COOLDOWN_SEC: float = _as_float_env("TRAIL_COOLDOWN_SEC", 12.0)
    TRAILING_ENABLE: bool = _as_bool(os.getenv("TRAILING_ENABLE", "true"))

    # Feature toggles
    ENABLE_LIVE_TRADING: bool = _as_bool(os.getenv("ENABLE_LIVE_TRADING", "false"))
    ENABLE_TELEGRAM: bool = _as_bool(os.getenv("ENABLE_TELEGRAM", "true"))
    ALLOW_OFFHOURS_TESTING: bool = _as_bool(os.getenv("ALLOW_OFFHOURS_TESTING", "false"))

    # Logging / persistence
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

    # Historical data timeframe
    HISTORICAL_TIMEFRAME: str = os.getenv("HISTORICAL_TIMEFRAME", "minute")

    # Session management
    SESSION_AUTO_EXIT_TIME: str = os.getenv("SESSION_AUTO_EXIT_TIME", "15:20")  # HH:MM IST


if __name__ == "__main__":
    import pprint
    print("Final Config:")
    pprint.pprint({
        k: v for k, v in Config.__dict__.items()
        if not k.startswith("__") and not callable(v)
    })
