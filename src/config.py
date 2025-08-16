# src/config.py

import os
from typing import List, Tuple, Any

def _to_bool(x: Any, default: bool = False) -> bool:
    if x is None:
        return default
    return str(x).strip().lower() in ("1", "true", "yes", "y", "on")

def _to_int(x: Any, default: int) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default

def _to_float(x: Any, default: float) -> float:
    try:
        return float(str(x).strip())
    except Exception:
        return default

def _to_list_csv(x: Any) -> List[str]:
    if not x:
        return []
    return [p.strip() for p in str(x).split(",") if p.strip()]

def _to_time_ranges(x: Any) -> List[Tuple[str, str]]:
    """Parse 'HH:MM-HH:MM,HH:MM-HH:MM' → [('HH:MM','HH:MM'), ...]"""
    ranges: List[Tuple[str, str]] = []
    for chunk in _to_list_csv(x):
        if "-" in chunk:
            a, b = chunk.split("-", 1)
            a = a.strip()
            b = b.strip()
            if len(a) == 5 and len(b) == 5:
                ranges.append((a, b))
    return ranges


class Config:
    # =========================
    # MODES / TOGGLES
    # =========================
    ENABLE_LIVE_TRADING = _to_bool(os.getenv("ENABLE_LIVE_TRADING"), False)
    ENABLE_TELEGRAM = _to_bool(os.getenv("ENABLE_TELEGRAM"), True)
    ALLOW_OFFHOURS_TESTING = _to_bool(os.getenv("ALLOW_OFFHOURS_TESTING"), False)
    USE_IST_CLOCK = _to_bool(os.getenv("USE_IST_CLOCK"), True)
    PREFERRED_EXIT_MODE = (os.getenv("PREFERRED_EXIT_MODE") or "REGULAR").upper()  # AUTO|GTT|REGULAR

    # --- Quality mode (runtime switch; kept for backward compatibility) ---
    QUALITY_MODE_DEFAULT = _to_bool(os.getenv("QUALITY_MODE_DEFAULT"), False)  # legacy: initial ON/OFF
    QUALITY_SCORE_BUMP = _to_float(os.getenv("QUALITY_SCORE_BUMP"), 1.0)

    # New: quality switch controller (parsed by trader & telegram)
    #   QUALITY_MODE: AUTO | ON | OFF
    QUALITY_MODE = (os.getenv("QUALITY_MODE") or ("ON" if QUALITY_MODE_DEFAULT else "AUTO")).upper()
    # Auto thresholds (trend ⇒ turn ON, range/noise ⇒ turn OFF)
    QUALITY_AUTO_ADX_MIN = _to_float(os.getenv("QUALITY_AUTO_ADX_MIN"), 20.0)     # turn ON above this
    QUALITY_AUTO_BB_WIDTH_MAX = _to_float(os.getenv("QUALITY_AUTO_BB_WIDTH_MAX"), 0.008)  # turn OFF if <= 0.8%
    QUALITY_AUTO_EVAL_MIN_BARS = _to_int(os.getenv("QUALITY_AUTO_EVAL_MIN_BARS"), 120)    # ~ last 2 hours on 1-min
    QUALITY_AUTO_HYSTERESIS_SEC = _to_int(os.getenv("QUALITY_AUTO_HYSTERESIS_SEC"), 90)   # avoid flip-flop
    QUALITY_VERBOSE_REASON = _to_bool(os.getenv("QUALITY_VERBOSE_REASON"), True)  # show reason in /status

    # Session auto-exit time (HH:MM IST)
    SESSION_AUTO_EXIT_TIME = os.getenv("SESSION_AUTO_EXIT_TIME", "15:20")

    # API keys (live)
    ZERODHA_API_KEY = os.getenv("ZERODHA_API_KEY") or ""
    KITE_ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN") or ""
    ZERODHA_ACCESS_TOKEN = os.getenv("ZERODHA_ACCESS_TOKEN") or ""

    # =========================
    # STRATEGY CORE
    # =========================
    MIN_SIGNAL_SCORE = _to_int(os.getenv("MIN_SIGNAL_SCORE"), 2)
    CONFIDENCE_THRESHOLD = _to_float(os.getenv("CONFIDENCE_THRESHOLD"), 5.2)
    BASE_STOP_LOSS_POINTS = _to_float(os.getenv("BASE_STOP_LOSS_POINTS"), 20.0)
    BASE_TARGET_POINTS = _to_float(os.getenv("BASE_TARGET_POINTS"), 40.0)
    ATR_PERIOD = _to_int(os.getenv("ATR_PERIOD"), 14)
    ATR_SL_MULTIPLIER = _to_float(os.getenv("ATR_SL_MULTIPLIER"), 1.5)
    ATR_TP_MULTIPLIER = _to_float(os.getenv("ATR_TP_MULTIPLIER"), 3.0)
    SL_CONFIDENCE_ADJ = _to_float(os.getenv("SL_CONFIDENCE_ADJ"), 0.12)
    TP_CONFIDENCE_ADJ = _to_float(os.getenv("TP_CONFIDENCE_ADJ"), 0.35)

    # Regime-aware (used by strategy and trader filters)
    REGIME_TREND_TP_MULT = _to_float(os.getenv("REGIME_TREND_TP_MULT"), 3.4)
    REGIME_TREND_SL_MULT = _to_float(os.getenv("REGIME_TREND_SL_MULT"), 1.6)
    REGIME_RANGE_TP_MULT = _to_float(os.getenv("REGIME_RANGE_TP_MULT"), 2.4)
    REGIME_RANGE_SL_MULT = _to_float(os.getenv("REGIME_RANGE_SL_MULT"), 1.3)
    REGIME_MRV_ENABLE = _to_bool(os.getenv("REGIME_MRV_ENABLE"), True)
    REGIME_MOM_ENABLE = _to_bool(os.getenv("REGIME_MOM_ENABLE"), True)

    # =========================
    # MTF & REGIME FILTERS (Trader gates)
    # =========================
    ENABLE_MTF_FILTER = _to_bool(os.getenv("ENABLE_MTF_FILTER"), True)
    MTF_TIMEFRAME = os.getenv("MTF_TIMEFRAME", "5minute")  # for Kite: "5minute"
    MTF_EMA_FAST = _to_int(os.getenv("MTF_EMA_FAST"), 21)
    MTF_EMA_SLOW = _to_int(os.getenv("MTF_EMA_SLOW"), 50)

    ENABLE_REGIME_FILTER = _to_bool(os.getenv("ENABLE_REGIME_FILTER"), True)
    REGIME_MODE = (os.getenv("REGIME_MODE") or "AUTO").upper()  # AUTO|TREND|RANGE|OFF
    ADX_PERIOD = _to_int(os.getenv("ADX_PERIOD"), 14)
    ADX_MIN_TREND = _to_float(os.getenv("ADX_MIN_TREND"), 18.0)
    BB_WINDOW = _to_int(os.getenv("BB_WINDOW"), 20)
    BB_WIDTH_MIN = _to_float(os.getenv("BB_WIDTH_MIN"), 0.006)   # 0.6%
    BB_WIDTH_MAX = _to_float(os.getenv("BB_WIDTH_MAX"), 0.02)    # 2%

    # Legacy HTF hints (kept for compatibility if strategy uses them)
    HTF_TIMEFRAME_MIN = _to_int(os.getenv("HTF_TIMEFRAME_MIN"), 5)
    HTF_EMA_PERIOD = _to_int(os.getenv("HTF_EMA_PERIOD"), 20)
    HTF_MIN_SLOPE = _to_float(os.getenv("HTF_MIN_SLOPE"), 0.0)

    # =========================
    # Warmup / data
    # =========================
    WARMUP_BARS = _to_int(os.getenv("WARMUP_BARS"), 25)
    DATA_LOOKBACK_MINUTES = _to_int(os.getenv("DATA_LOOKBACK_MINUTES"), 45)
    HISTORICAL_TIMEFRAME = os.getenv("HISTORICAL_TIMEFRAME", "minute")
    SKIP_FIRST_MIN = _to_int(os.getenv("SKIP_FIRST_MIN"), 5)

    # =========================
    # INSTRUMENTS / LOTS
    # =========================
    SPOT_SYMBOL = os.getenv("SPOT_SYMBOL", "NSE:NIFTY 50")
    TRADE_SYMBOL = os.getenv("TRADE_SYMBOL", "NIFTY")
    TRADE_EXCHANGE = os.getenv("TRADE_EXCHANGE", "NFO")
    INSTRUMENT_TOKEN = _to_int(os.getenv("INSTRUMENT_TOKEN"), 256265)  # NIFTY spot token
    NIFTY_LOT_SIZE = _to_int(os.getenv("NIFTY_LOT_SIZE"), 75)
    MIN_LOTS = _to_int(os.getenv("MIN_LOTS"), 1)
    MAX_LOTS = _to_int(os.getenv("MAX_LOTS"), 15)
    STRIKE_RANGE = _to_int(os.getenv("STRIKE_RANGE"), 3)

    # =========================
    # RISK / CIRCUIT BREAKERS
    # =========================
    RISK_PER_TRADE = _to_float(os.getenv("RISK_PER_TRADE"), 0.025)
    MAX_DRAWDOWN = _to_float(os.getenv("MAX_DRAWDOWN"), 0.07)
    CONSECUTIVE_LOSS_LIMIT = _to_int(os.getenv("CONSECUTIVE_LOSS_LIMIT"), 3)
    MAX_DAILY_DRAWDOWN_PCT = _to_float(os.getenv("MAX_DAILY_DRAWDOWN_PCT"), 0.05)
    CIRCUIT_RELEASE_PCT = _to_float(os.getenv("CIRCUIT_RELEASE_PCT"), 0.015)
    HALT_ON_DRAWDOWN = _to_bool(os.getenv("HALT_ON_DRAWDOWN"), True)
    MAX_CONCURRENT_POSITIONS = _to_int(os.getenv("MAX_CONCURRENT_POSITIONS"), 1)
    MAX_TRADES_PER_DAY = _to_int(os.getenv("MAX_TRADES_PER_DAY"), 30)
    LOSS_COOLDOWN_MIN = _to_int(os.getenv("LOSS_COOLDOWN_MIN"), 2)

    # Streak manager
    LOSS_STREAK_HALVE_SIZE = _to_int(os.getenv("LOSS_STREAK_HALVE_SIZE"), 3)
    LOSS_STREAK_PAUSE_MIN = _to_int(os.getenv("LOSS_STREAK_PAUSE_MIN"), 20)

    # Session R ladders
    DAY_STOP_AFTER_POS_R = _to_float(os.getenv("DAY_STOP_AFTER_POS_R"), 4.0)
    DAY_HALF_SIZE_AFTER_POS_R = _to_float(os.getenv("DAY_HALF_SIZE_AFTER_POS_R"), 2.0)
    DAY_STOP_AFTER_NEG_R = _to_float(os.getenv("DAY_STOP_AFTER_NEG_R"), -3.0)

    # Volatility-scaled risk
    ATR_TARGET = _to_float(os.getenv("ATR_TARGET"), 10.0)
    ATR_MIN = _to_float(os.getenv("ATR_MIN"), 3.0)
    VOL_RISK_CLAMP_MIN = _to_float(os.getenv("VOL_RISK_CLAMP_MIN"), 0.5)
    VOL_RISK_CLAMP_MAX = _to_float(os.getenv("VOL_RISK_CLAMP_MAX"), 1.5)

    # =========================
    # OPTIONS FILTERS
    # =========================
    OPTION_TYPE = (os.getenv("OPTION_TYPE") or "BOTH").upper()
    OPTION_SL_PERCENT = _to_float(os.getenv("OPTION_SL_PERCENT"), 0.05)
    OPTION_TP_PERCENT = _to_float(os.getenv("OPTION_TP_PERCENT"), 0.20)
    OPTION_BREAKOUT_PCT = _to_float(os.getenv("OPTION_BREAKOUT_PCT"), 0.003)
    OPTION_SPOT_TREND_PCT = _to_float(os.getenv("OPTION_SPOT_TREND_PCT"), 0.0025)
    OPTION_REQUIRE_SPOT_CONFIRM = _to_bool(os.getenv("OPTION_REQUIRE_SPOT_CONFIRM"), False)

    MIN_PREMIUM = _to_float(os.getenv("MIN_PREMIUM"), 12.0)
    SPREAD_MEAN_PCT_MAX = _to_float(os.getenv("SPREAD_MEAN_PCT_MAX"), 0.03)
    IV_SOURCE = (os.getenv("IV_SOURCE") or "LTP_IMPLIED").upper()
    RISK_FREE_RATE = _to_float(os.getenv("RISK_FREE_RATE"), 0.06)
    IV_MAX = _to_float(os.getenv("IV_MAX"), 0.70)
    EXPIRY_PREFERENCE = (os.getenv("EXPIRY_PREFERENCE") or "NEAR").upper()
    REQUIRE_OI = _to_bool(os.getenv("REQUIRE_OI"), False)
    MIN_OI = _to_int(os.getenv("MIN_OI"), 30000)

    # =========================
    # EXECUTION / WORKERS
    # =========================
    DEFAULT_PRODUCT = os.getenv("DEFAULT_PRODUCT", "MIS")
    DEFAULT_ORDER_TYPE = (os.getenv("DEFAULT_ORDER_TYPE") or "MARKET").upper()
    DEFAULT_ORDER_TYPE_EXIT = (os.getenv("DEFAULT_ORDER_TYPE_EXIT") or "LIMIT").upper()
    DEFAULT_VALIDITY = (os.getenv("DEFAULT_VALIDITY") or "DAY").upper()
    TICK_SIZE = _to_float(os.getenv("TICK_SIZE"), 0.05)

    TRAILING_ENABLE = _to_bool(os.getenv("TRAILING_ENABLE"), True)
    TRAIL_COOLDOWN_SEC = _to_float(os.getenv("TRAIL_COOLDOWN_SEC"), 12.0)
    TRAIL_MIN_POINTS = _to_float(os.getenv("TRAIL_MIN_POINTS"), 1.0)
    WORKER_INTERVAL_SEC = _to_int(os.getenv("WORKER_INTERVAL_SEC"), 10)
    NFO_FREEZE_QTY = _to_int(os.getenv("NFO_FREEZE_QTY"), 1800)

    MAX_ENTRY_SLIP_BPS = _to_float(os.getenv("MAX_ENTRY_SLIP_BPS"), 25.0)
    ORDER_RETRY_LIMIT = _to_int(os.getenv("ORDER_RETRY_LIMIT"), 2)
    ORDER_RETRY_TICK_OFFSET = _to_int(os.getenv("ORDER_RETRY_TICK_OFFSET"), 2)

    # Spread guard
    SPREAD_GUARD_MODE = (os.getenv("SPREAD_GUARD_MODE") or "LTP_MID").upper()  # RANGE|LTP_MID
    SPREAD_GUARD_BA_MAX = _to_float(os.getenv("SPREAD_GUARD_BA_MAX"), 0.015)
    SPREAD_GUARD_LTPMID_MAX = _to_float(os.getenv("SPREAD_GUARD_LTPMID_MAX"), 0.015)
    SPREAD_GUARD_PCT = _to_float(os.getenv("SPREAD_GUARD_PCT"), 0.03)
    DYNAMIC_SPREAD_GUARD = _to_bool(os.getenv("DYNAMIC_SPREAD_GUARD"), True)
    SPREAD_VOL_LOOKBACK = _to_int(os.getenv("SPREAD_VOL_LOOKBACK"), 20)

    # =========================
    # EXITS / PARTIALS
    # =========================
    PARTIAL_TP_ENABLE = _to_bool(os.getenv("PARTIAL_TP_ENABLE"), True)
    PARTIAL_TP_RATIO = _to_float(os.getenv("PARTIAL_TP_RATIO"), 0.50)
    PARTIAL_TP_USE_MIDPOINT = _to_bool(os.getenv("PARTIAL_TP_USE_MIDPOINT"), True)
    PARTIAL_TP2_R_MULT = _to_float(os.getenv("PARTIAL_TP2_R_MULT"), 2.0)

    BREAKEVEN_AFTER_TP1_ENABLE = _to_bool(os.getenv("BREAKEVEN_AFTER_TP1_ENABLE"), True)
    BREAKEVEN_OFFSET_TICKS = _to_int(os.getenv("BREAKEVEN_OFFSET_TICKS"), 1)

    HARD_STOP_ENABLE = _to_bool(os.getenv("HARD_STOP_ENABLE"), True)
    HARD_STOP_GRACE_SEC = _to_float(os.getenv("HARD_STOP_GRACE_SEC"), 3.0)
    HARD_STOP_SLIPPAGE_BPS = _to_float(os.getenv("HARD_STOP_SLIPPAGE_BPS"), 5.0)

    CHANDELIER_N = _to_int(os.getenv("CHANDELIER_N"), 22)
    CHANDELIER_K = _to_float(os.getenv("CHANDELIER_K"), 2.5)
    VWAP_TRAIL_ENABLE = _to_bool(os.getenv("VWAP_TRAIL_ENABLE"), True)

    MAX_HOLD_MIN = _to_int(os.getenv("MAX_HOLD_MIN"), 25)
    BOX_HOLD_PCT = _to_float(os.getenv("BOX_HOLD_PCT"), 0.01)

    # =========================
    # SCHEDULING / HOURS
    # =========================
    PEAK_POLL_SEC = _to_int(os.getenv("PEAK_POLL_SEC"), 12)
    OFFPEAK_POLL_SEC = _to_int(os.getenv("OFFPEAK_POLL_SEC"), 25)
    TIME_FILTER_START = os.getenv("TIME_FILTER_START", "09:20")
    TIME_FILTER_END = os.getenv("TIME_FILTER_END", "15:20")

    ENABLE_TIME_BUCKETS = _to_bool(os.getenv("ENABLE_TIME_BUCKETS"), False)
    TIME_BUCKETS = _to_time_ranges(os.getenv("TIME_BUCKETS", ""))

    ENABLE_EVENT_WINDOWS = _to_bool(os.getenv("ENABLE_EVENT_WINDOWS"), False)
    EVENT_WINDOWS = _to_time_ranges(os.getenv("EVENT_WINDOWS", ""))

    # =========================
    # FEES / SLIPPAGE
    # =========================
    SLIPPAGE_BPS = _to_float(os.getenv("SLIPPAGE_BPS"), 4.0)
    FEES_PER_LOT = _to_float(os.getenv("FEES_PER_LOT"), 25.0)
    # Used by RealTimeTrader P&L calc: percent-per-side (e.g., 0.03 = 0.03%)
    FEES_PCT_PER_SIDE = _to_float(os.getenv("FEES_PCT_PER_SIDE"), 0.03)

    # =========================
    # LOGGING / DIAGNOSTICS
    # =========================
    LOG_FILE = os.getenv("LOG_FILE", "logs/trades.csv")
    DIAGNOSTIC_VERBOSE = _to_bool(os.getenv("DIAGNOSTIC_VERBOSE"), False)
    LOG_LEVEL = (os.getenv("LOG_LEVEL") or "DEBUG").upper()
    BALANCE_LOG_INTERVAL_MIN = _to_int(os.getenv("BALANCE_LOG_INTERVAL_MIN"), 30)
    DETAIL_LOGS_ENABLE = _to_bool(os.getenv("DETAIL_LOGS_ENABLE"), True)
    HEARTBEAT_MIN = _to_int(os.getenv("HEARTBEAT_MIN"), 3)

    # =========================
    # TELEGRAM
    # =========================
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN") or ""
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID") or ""

    # =========================
    # ORDER SAFETY / STATE
    # =========================
    IDEMP_TTL_SEC = _to_int(os.getenv("IDEMP_TTL_SEC"), 60)
    PERSIST_REATTACH_ON_START = _to_bool(os.getenv("PERSIST_REATTACH_ON_START"), True)
