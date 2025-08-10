# src/config.py
"""
Centralised configuration loader for the scalper bot.
- Loads .env if present
- Robust parsing (bool/int/float/aliases)
- Backward compatibility: accepts both ZERODHA_ACCESS_TOKEN and KITE_ACCESS_TOKEN
"""

from __future__ import annotations
import os
from pathlib import Path

def _to_bool(v: str | None, default=False) -> bool:
    if v is None:
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")

def _to_int(v: str | None, default=0) -> int:
    try:
        return int(float(str(v).strip()))
    except Exception:
        return default

def _to_float(v: str | None, default=0.0) -> float:
    try:
        return float(str(v).strip())
    except Exception:
        return default

# Load .env (optional)
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        print(f"🔐 Loaded environment from {env_path}")
    else:
        print(f"⚠️  .env not found at: {env_path}")
except Exception as exc:
    print(f"⚠️  dotenv load skipped: {exc}")

class Config:
    # ---------------- Zerodha ----------------
    ZERODHA_API_KEY: str = os.getenv("ZERODHA_API_KEY", "").strip()
    ZERODHA_API_SECRET: str = os.getenv("ZERODHA_API_SECRET", "").strip()

    # Accept BOTH names for the daily token
    _ACCESS_TOKEN_A = os.getenv("ZERODHA_ACCESS_TOKEN", "").strip()
    _ACCESS_TOKEN_B = os.getenv("KITE_ACCESS_TOKEN", "").strip()
    ZERODHA_ACCESS_TOKEN: str = _ACCESS_TOKEN_A or _ACCESS_TOKEN_B

    # ---------------- Telegram ---------------
    TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    TELEGRAM_CHAT_ID: int = _to_int(os.getenv("TELEGRAM_CHAT_ID"), 0)

    # ---------------- Risk & Strategy --------
    RISK_PER_TRADE: float = _to_float(os.getenv("RISK_PER_TRADE"), 0.01)
    MAX_DRAWDOWN: float = _to_float(os.getenv("MAX_DRAWDOWN"), 0.05)
    CONSECUTIVE_LOSS_LIMIT: int = _to_int(os.getenv("CONSECUTIVE_LOSS_LIMIT"), 3)

    BASE_STOP_LOSS_POINTS: float = _to_float(os.getenv("BASE_STOP_LOSS_POINTS"), 20.0)
    BASE_TARGET_POINTS: float = _to_float(os.getenv("BASE_TARGET_POINTS"), 40.0)
    CONFIDENCE_THRESHOLD: float = _to_float(os.getenv("CONFIDENCE_THRESHOLD"), 8.0)
    MIN_SIGNAL_SCORE: int = _to_int(os.getenv("MIN_SIGNAL_SCORE"), 7)

    TIME_FILTER_START: str = os.getenv("TIME_FILTER_START", "09:15")
    TIME_FILTER_END: str = os.getenv("TIME_FILTER_END", "15:15")

    ATR_SL_MULTIPLIER: float = _to_float(os.getenv("ATR_SL_MULTIPLIER"), 1.5)
    ATR_TP_MULTIPLIER: float = _to_float(os.getenv("ATR_TP_MULTIPLIER"), 3.0)

    SL_CONFIDENCE_ADJ: float = _to_float(os.getenv("SL_CONFIDENCE_ADJ"), 0.2)
    TP_CONFIDENCE_ADJ: float = _to_float(os.getenv("TP_CONFIDENCE_ADJ"), 0.3)

    # ---------------- Instruments ------------
    NIFTY_LOT_SIZE: int = _to_int(os.getenv("NIFTY_LOT_SIZE"), 75)
    MIN_LOTS: int = _to_int(os.getenv("MIN_LOTS"), 1)
    MAX_LOTS: int = _to_int(os.getenv("MAX_LOTS"), 5)

    TRADE_SYMBOL: str = os.getenv("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = os.getenv("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN: int = _to_int(os.getenv("INSTRUMENT_TOKEN"), 256265)

    SPOT_SYMBOL: str = os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50")
    OPTION_TYPE: str = os.getenv("OPTION_TYPE", "BOTH")
    STRIKE_SELECTION_TYPE: str = os.getenv("STRIKE_SELECTION_TYPE", "ATM")
    STRIKE_RANGE: int = _to_int(os.getenv("STRIKE_RANGE"), 4)
    DATA_LOOKBACK_MINUTES: int = _to_int(os.getenv("DATA_LOOKBACK_MINUTES"), 35)

    OPTION_SL_PERCENT: float = _to_float(os.getenv("OPTION_SL_PERCENT"), 0.05)
    OPTION_TP_PERCENT: float = _to_float(os.getenv("OPTION_TP_PERCENT"), 0.15)
    OPTION_BREAKOUT_PCT: float = _to_float(os.getenv("OPTION_BREAKOUT_PCT"), 0.01)
    OPTION_SPOT_TREND_PCT: float = _to_float(os.getenv("OPTION_SPOT_TREND_PCT"), 0.005)

    # ---------------- Execution --------------
    DEFAULT_PRODUCT: str = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = os.getenv("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = os.getenv("DEFAULT_VALIDITY", "DAY")

    # Exit behavior / trailing
    TICK_SIZE: float = _to_float(os.getenv("TICK_SIZE"), 0.05)
    TRAIL_COOLDOWN_SEC: float = _to_float(os.getenv("TRAIL_COOLDOWN_SEC"), 12.0)
    PREFERRED_EXIT_MODE: str = os.getenv("PREFERRED_EXIT_MODE", "AUTO")  # AUTO|GTT|REGULAR

    # ---------------- Toggles ----------------
    ENABLE_LIVE_TRADING: bool = _to_bool(os.getenv("ENABLE_LIVE_TRADING"), False)
    ENABLE_TELEGRAM: bool = _to_bool(os.getenv("ENABLE_TELEGRAM"), True)

    # ---------------- Logging ----------------
    LOG_FILE: str = os.getenv("LOG_FILE", "logs/trades.csv")

if __name__ == "__main__":
    # Quick view (do not print secrets)
    from pprint import pprint
    safe = {k: v for k, v in Config.__dict__.items()
            if not k.startswith("__") and not callable(v)}
    safe["ZERODHA_API_SECRET"] = "***"
    safe["ZERODHA_ACCESS_TOKEN"] = "***" if safe.get("ZERODHA_ACCESS_TOKEN") else ""
    pprint(safe)