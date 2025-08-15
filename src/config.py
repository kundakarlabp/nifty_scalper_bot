# src/config.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple

# Optional: load .env if present (no hard dependency if you don't use python-dotenv)
try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()  # loads from .env in project root if available
except Exception:
    pass


# ----------------------------- helpers ----------------------------- #

def _get_str(key: str, default: str = "") -> str:
    v = os.getenv(key)
    return v if v is not None and v != "" else default


def _get_bool(key: str, default: bool = False) -> bool:
    v = os.getenv(key)
    if v is None or v == "":
        return default
    return str(v).strip().lower() in ("1", "true", "yes", "y", "on")


def _get_int(key: str, default: int = 0, *, minv: Optional[int] = None, maxv: Optional[int] = None) -> int:
    v = os.getenv(key)
    try:
        x = int(float(v)) if v is not None and v != "" else default  # tolerate "5.0"
    except Exception:
        x = default
    if minv is not None:
        x = max(minv, x)
    if maxv is not None:
        x = min(maxv, x)
    return x


def _get_float(key: str, default: float = 0.0, *, minv: Optional[float] = None, maxv: Optional[float] = None) -> float:
    v = os.getenv(key)
    try:
        x = float(v) if v is not None and v != "" else default
    except Exception:
        x = default
    if minv is not None:
        x = max(minv, x)
    if maxv is not None:
        x = min(maxv, x)
    return x


def _get_list(key: str, default: List[str] | None = None, sep: str = ",") -> List[str]:
    v = os.getenv(key)
    if v is None or str(v).strip() == "":
        return list(default or [])
    return [s.strip() for s in str(v).split(sep) if s.strip()]


def _get_time_hhmm(key: str, default: str) -> Tuple[int, int]:
    """Returns (hour, minute) with basic validation."""
    v = _get_str(key, default)
    try:
        hh, mm = v.split(":")
        h, m = int(hh), int(mm)
        h = max(0, min(23, h))
        m = max(0, min(59, m))
        return h, m
    except Exception:
        dh, dm = default.split(":")
        return int(dh), int(dm)


# ----------------------------- config object ----------------------------- #

@dataclass(frozen=True)
class Config:
    # ===== MODES / TOGGLES =====
    ENABLE_LIVE_TRADING: bool = _get_bool("ENABLE_LIVE_TRADING", True)
    ENABLE_TELEGRAM: bool = _get_bool("ENABLE_TELEGRAM", True)
    ALLOW_OFFHOURS_TESTING: bool = _get_bool("ALLOW_OFFHOURS_TESTING", False)
    PREFERRED_EXIT_MODE: str = _get_str("PREFERRED_EXIT_MODE", "REGULAR").upper()  # AUTO|GTT|REGULAR
    USE_SLM_EXIT: bool = _get_bool("USE_SLM_EXIT", True)
    SL_LIMIT_OFFSET_TICKS: int = _get_int("SL_LIMIT_OFFSET_TICKS", 2, minv=0)

    # ===== STRATEGY CORE =====
    MIN_SIGNAL_SCORE: int = _get_int("MIN_SIGNAL_SCORE", 2, minv=1)
    CONFIDENCE_THRESHOLD: float = _get_float("CONFIDENCE_THRESHOLD", 5.2, minv=1.0, maxv=10.0)
    BASE_STOP_LOSS_POINTS: float = _get_float("BASE_STOP_LOSS_POINTS", 20.0, minv=0.5)
    BASE_TARGET_POINTS: float = _get_float("BASE_TARGET_POINTS", 40.0, minv=0.5)
    ATR_PERIOD: int = _get_int("ATR_PERIOD", 14, minv=2)
    ATR_SL_MULTIPLIER: float = _get_float("ATR_SL_MULTIPLIER", 1.5, minv=0.1)
    ATR_TP_MULTIPLIER: float = _get_float("ATR_TP_MULTIPLIER", 3.0, minv=0.1)
    SL_CONFIDENCE_ADJ: float = _get_float("SL_CONFIDENCE_ADJ", 0.12, minv=0.0)
    TP_CONFIDENCE_ADJ: float = _get_float("TP_CONFIDENCE_ADJ", 0.35, minv=0.0)

    # ===== DATA / WARMUP =====
    WARMUP_BARS: int = _get_int("WARMUP_BARS", 25, minv=5)
    DATA_LOOKBACK_MINUTES: int = _get_int("DATA_LOOKBACK_MINUTES", 45, minv=10)
    HISTORICAL_TIMEFRAME: str = _get_str("HISTORICAL_TIMEFRAME", "minute")
    SKIP_FIRST_MIN: int = _get_int("SKIP_FIRST_MIN", 5, minv=0, maxv=30)

    # ===== INSTRUMENTS =====
    SPOT_SYMBOL: str = _get_str("SPOT_SYMBOL", "NSE:NIFTY 50")
    TRADE_SYMBOL: str = _get_str("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE: str = _get_str("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN: int = _get_int("INSTRUMENT_TOKEN", 256265)
    NIFTY_LOT_SIZE: int = _get_int("NIFTY_LOT_SIZE", 75, minv=1)
    MIN_LOTS: int = _get_int("MIN_LOTS", 1, minv=1)
    MAX_LOTS: int = _get_int("MAX_LOTS", 15, minv=1)
    STRIKE_RANGE: int = _get_int("STRIKE_RANGE", 3, minv=0)

    # ===== RISK / CIRCUIT BREAKERS =====
    RISK_PER_TRADE: float = _get_float("RISK_PER_TRADE", 0.025, minv=0.001, maxv=0.2)
    MAX_DRAWDOWN: float = _get_float("MAX_DRAWDOWN", 0.07, minv=0.01, maxv=0.99)
    CONSECUTIVE_LOSS_LIMIT: int = _get_int("CONSECUTIVE_LOSS_LIMIT", 3, minv=1)
    MAX_DAILY_DRAWDOWN_PCT: float = _get_float("MAX_DAILY_DRAWDOWN_PCT", 0.05, minv=0.005, maxv=0.99)
    CIRCUIT_RELEASE_PCT: float = _get_float("CIRCUIT_RELEASE_PCT", 0.015, minv=0.001)
    HALT_ON_DRAWDOWN: bool = _get_bool("HALT_ON_DRAWDOWN", True)
    MAX_CONCURRENT_POSITIONS: int = _get_int("MAX_CONCURRENT_POSITIONS", 1, minv=1)
    MAX_TRADES_PER_DAY: int = _get_int("MAX_TRADES_PER_DAY", 30, minv=1)

    # ===== OPTIONS BREAKOUT FILTERS =====
    OPTION_TYPE: str = _get_str("OPTION_TYPE", "BOTH").upper()  # CE|PE|BOTH
    OPTION_SL_PERCENT: float = _get_float("OPTION_SL_PERCENT", 0.05, minv=0.001, maxv=1.0)
    OPTION_TP_PERCENT: float = _get_float("OPTION_TP_PERCENT", 0.20, minv=0.001, maxv=5.0)
    OPTION_BREAKOUT_PCT: float = _get_float("OPTION_BREAKOUT_PCT", 0.003, minv=0.0001)
    OPTION_SPOT_TREND_PCT: float = _get_float("OPTION_SPOT_TREND_PCT", 0.0025, minv=0.0)
    OPTION_REQUIRE_SPOT_CONFIRM: bool = _get_bool("OPTION_REQUIRE_SPOT_CONFIRM", False)

    # ===== EXECUTION / WORKERS =====
    DEFAULT_PRODUCT: str = _get_str("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE: str = _get_str("DEFAULT_ORDER_TYPE", "MARKET")
    DEFAULT_VALIDITY: str = _get_str("DEFAULT_VALIDITY", "DAY")
    TICK_SIZE: float = _get_float("TICK_SIZE", 0.05, minv=0.01)
    TRAILING_ENABLE: bool = _get_bool("TRAILING_ENABLE", True)
    TRAIL_COOLDOWN_SEC: float = _get_float("TRAIL_COOLDOWN_SEC", 8.0, minv=1.0)
    WORKER_INTERVAL_SEC: int = _get_int("WORKER_INTERVAL_SEC", 4, minv=1)
    NFO_FREEZE_QTY: int = _get_int("NFO_FREEZE_QTY", 1800, minv=1)

    # ===== SPREAD / MICROSTRUCTURE GUARDRAILS =====
    SPREAD_GUARD_MODE: str = _get_str("SPREAD_GUARD_MODE", "RANGE").upper()  # RANGE|LTP_MID
    SPREAD_GUARD_BA_MAX: float = _get_float("SPREAD_GUARD_BA_MAX", 0.015, minv=0.0001)
    SPREAD_GUARD_LTPMID_MAX: float = _get_float("SPREAD_GUARD_LTPMID_MAX", 0.015, minv=0.0001)
    SPREAD_GUARD_PCT: float = _get_float("SPREAD_GUARD_PCT", 0.03, minv=0.0001)

    # ===== FEES / SLIPPAGE =====
    SLIPPAGE_BPS: float = _get_float("SLIPPAGE_BPS", 4.0, minv=0.0)
    FEES_PER_LOT: float = _get_float("FEES_PER_LOT", 25.0, minv=0.0)

    # ===== PARTIAL TP / BREAKEVEN / HARD STOP =====
    PARTIAL_TP_ENABLE: bool = _get_bool("PARTIAL_TP_ENABLE", True)
    PARTIAL_TP_RATIO: float = _get_float("PARTIAL_TP_RATIO", 0.50, minv=0.05, maxv=0.95)
    PARTIAL_TP_USE_MIDPOINT: bool = _get_bool("PARTIAL_TP_USE_MIDPOINT", True)
    BREAKEVEN_AFTER_TP1_ENABLE: bool = _get_bool("BREAKEVEN_AFTER_TP1_ENABLE", True)
    BREAKEVEN_OFFSET_TICKS: int = _get_int("BREAKEVEN_OFFSET_TICKS", 1, minv=0)
    HARD_STOP_ENABLE: bool = _get_bool("HARD_STOP_ENABLE", True)
    HARD_STOP_GRACE_SEC: float = _get_float("HARD_STOP_GRACE_SEC", 3.0, minv=0.0)
    HARD_STOP_SLIPPAGE_BPS: float = _get_float("HARD_STOP_SLIPPAGE_BPS", 6.0, minv=0.0)
    PARTIAL_TP2_R_MULT: float = _get_float("PARTIAL_TP2_R_MULT", 2.0, minv=1.0)

    # ===== LOGGING =====
    LOG_FILE: str = _get_str("LOG_FILE", "logs/trades.csv")
    DIAGNOSTIC_VERBOSE: bool = _get_bool("DIAGNOSTIC_VERBOSE", False)
    LOG_LEVEL: str = _get_str("LOG_LEVEL", "INFO").upper()
    BALANCE_LOG_INTERVAL_MIN: int = _get_int("BALANCE_LOG_INTERVAL_MIN", 30, minv=1)

    # ===== HOURS (IST) =====
    USE_IST_CLOCK: bool = _get_bool("USE_IST_CLOCK", True)
    TIME_FILTER_START_HM: Tuple[int, int] = _get_time_hhmm("TIME_FILTER_START", "09:20")
    TIME_FILTER_END_HM: Tuple[int, int] = _get_time_hhmm("TIME_FILTER_END", "15:20")

    # ===== STRIKE SELECTION WITH GREEKS =====
    USE_GREEKS_STRIKE_RANKING: bool = _get_bool("USE_GREEKS_STRIKE_RANKING", True)
    TARGET_DELTA_CALL: float = _get_float("TARGET_DELTA_CALL", 0.35)
    TARGET_DELTA_PUT: float = _get_float("TARGET_DELTA_PUT", -0.35)
    DELTA_TOL: float = _get_float("DELTA_TOL", 0.07, minv=0.0)
    REQUIRE_OI: bool = _get_bool("REQUIRE_OI", False)
    MIN_OI: int = _get_int("MIN_OI", 30000, minv=0)
    RISK_FREE_RATE: float = _get_float("RISK_FREE_RATE", 0.06)
    IV_SOURCE: str = _get_str("IV_SOURCE", "LTP_IMPLIED").upper()
    EXPIRY_PREFERENCE: str = _get_str("EXPIRY_PREFERENCE", "NEAR").upper()

    # ===== ADAPTIVE SCHEDULING / LIMITS =====
    PEAK_POLL_SEC: int = _get_int("PEAK_POLL_SEC", 12, minv=2)
    OFFPEAK_POLL_SEC: int = _get_int("OFFPEAK_POLL_SEC", 25, minv=3)
    LOSS_COOLDOWN_MIN: int = _get_int("LOSS_COOLDOWN_MIN", 2, minv=0)
    PREFERRED_TIE_RULE: str = _get_str("PREFERRED_TIE_RULE", "TREND").upper()

    # ===== NEW: MTF & ENTRY FILTERS =====
    MTF_CONFIRM_ENABLE: bool = _get_bool("MTF_CONFIRM_ENABLE", False)
    MTF_HIGHER_TF: str = _get_str("MTF_HIGHER_TF", "5min")
    MTF_CONFIRM_RULE: str = _get_str("MTF_CONFIRM_RULE", "EMA/SUPERTREND").upper()

    PULLBACK_ENTRY_ENABLE: bool = _get_bool("PULLBACK_ENTRY_ENABLE", False)
    PULLBACK_TYPES: List[str] = tuple(_get_list("PULLBACK_TYPES", ["RSI", "VWAP", "BB"]))

    PULLBACK_LOOKBACK: int = _get_int("PULLBACK_LOOKBACK", 7, minv=1)
    PULLBACK_MIN_BOUNCE_PCT: float = _get_float("PULLBACK_MIN_BOUNCE_PCT", 0.20, minv=0.0)
    PULLBACK_MAX_BB_TOUCHES: int = _get_int("PULLBACK_MAX_BB_TOUCHES", 2, minv=0)

    # ===== NEW: EXECUTION GUARDS =====
    EXECUTION_GUARDS_ENABLE: bool = _get_bool("EXECUTION_GUARDS_ENABLE", True)
    LTP_DEPTH_ENABLE: bool = _get_bool("LTP_DEPTH_ENABLE", False)
    LTP_DEPTH_MAX_SKEW_PCT: float = _get_float("LTP_DEPTH_MAX_SKEW_PCT", 0.30, minv=0.0)
    LTP_DEPTH_MIN_QTY: int = _get_int("LTP_DEPTH_MIN_QTY", 150, minv=0)

    # ===== NEW: PROFIT-LOCK LADDER =====
    PROFIT_LOCK_ENABLE: bool = _get_bool("PROFIT_LOCK_ENABLE", True)
    PROFIT_LOCK_STEPS: int = _get_int("PROFIT_LOCK_STEPS", 3, minv=0, maxv=10)
    PROFIT_LOCK_STEP_TICKS: int = _get_int("PROFIT_LOCK_STEP_TICKS", 5, minv=1)
    PROFIT_LOCK_STEP_SL_BACK_TICKS: int = _get_int("PROFIT_LOCK_STEP_SL_BACK_TICKS", 2, minv=0)

    # ===== NEW: REGIME SPLIT =====
    REGIME_SPLIT_ENABLE: bool = _get_bool("REGIME_SPLIT_ENABLE", True)
    REGIME_ADX_TREND: int = _get_int("REGIME_ADX_TREND", 22, minv=5, maxv=60)
    REGIME_BBWIDTH_TREND: float = _get_float("REGIME_BBWIDTH_TREND", 0.018, minv=0.0)
    REGIME_ATR_MIN: float = _get_float("REGIME_ATR_MIN", 5.0, minv=0.0)
    REGIME_WARMUP_BARS: int = _get_int("REGIME_WARMUP_BARS", 30, minv=5)

    TREND_TP_MULT: float = _get_float("TREND_TP_MULT", 3.4, minv=0.5)
    TREND_SL_MULT: float = _get_float("TREND_SL_MULT", 1.4, minv=0.1)
    RANGE_TP_MULT: float = _get_float("RANGE_TP_MULT", 2.4, minv=0.5)
    RANGE_SL_MULT: float = _get_float("RANGE_SL_MULT", 1.2, minv=0.1)

    COOLDOWN_AFTER_LOSS_MIN: int = _get_int("COOLDOWN_AFTER_LOSS_MIN", 2, minv=0)
    COOLDOWN_AFTER_WIN_MIN: int = _get_int("COOLDOWN_AFTER_WIN_MIN", 0, minv=0)

    # ===== OPTIONAL: SECRETS (do not print) =====
    ZERODHA_API_KEY: str = _get_str("ZERODHA_API_KEY", "")
    KITE_ACCESS_TOKEN: str = _get_str("KITE_ACCESS_TOKEN", "")
    ZERODHA_ACCESS_TOKEN: str = _get_str("ZERODHA_ACCESS_TOKEN", "")
    TELEGRAM_BOT_TOKEN: str = _get_str("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_ID: str = _get_str("TELEGRAM_CHAT_ID", "")

    # -------- convenience accessors --------
    @staticmethod
    def market_time_window() -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """(start_h, start_m), (end_h, end_m) in IST by default."""
        return Config.TIME_FILTER_START_HM, Config.TIME_FILTER_END_HM

    @staticmethod
    def has_live_creds() -> bool:
        return bool(Config.ZERODHA_API_KEY and (Config.KITE_ACCESS_TOKEN or Config.ZERODHA_ACCESS_TOKEN))

    @staticmethod
    def has_telegram() -> bool:
        return bool(Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID)