"""
===============================================================================
NEW FILE: src/nifty_scalper_bot/utils/market_hours.py
===============================================================================

MARKET HOURS UTILITY - Single Source of Truth
This module provides centralized market hours checking for the entire bot.

CREATE THIS FILE AT: src/nifty_scalper_bot/utils/market_hours.py
===============================================================================
"""

from __future__ import annotations

import os
from datetime import datetime, time as dtime
from enum import Enum
from functools import lru_cache
from typing import Tuple
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.logging import get_logger

LOGGER = get_logger(__name__)

# Indian Standard Time
IST = ZoneInfo("Asia/Kolkata")

# Market timing constants
def _env_truthy(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }


def _env_time(name: str, default: dtime) -> dtime:
    raw = os.getenv(name)
    if not raw:
        return default
    try:
        hour_s, minute_s = raw.strip().split(":", 1)
        hour = int(hour_s)
        minute = int(minute_s)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return default
        return dtime(hour, minute)
    except Exception:
        return default


# Market timing constants
MARKET_OPEN = _env_time("MARKET_OPEN_TIME", dtime(9, 15))
SAFE_START = _env_time("SAFE_START_TIME", dtime(9, 20))
SAFE_END = _env_time("SAFE_END_TIME", dtime(15, 25))
MARKET_CLOSE = _env_time("MARKET_CLOSE_TIME", dtime(15, 30))


class MarketState(Enum):
    """Market session state classification."""

    CLOSED = 'closed'
    PREOPEN = 'preopen'
    OPEN = 'open'
    POSTMARKET = 'postmarket'


def _override_enabled() -> bool:
    """Args: none; Returns: bool; Raises: none."""
    return allow_offhours_testing_safe()


def allow_offhours_testing_safe() -> bool:
    """Args: none; Returns: safe off-hours override state; Raises: none."""
    execution_mode = os.getenv("EXECUTION_MODE", "SHADOW").strip().upper()
    live_enabled = _env_truthy("ENABLE_LIVE") or _env_truthy("ENABLE_LIVE_TRADING")
    if execution_mode == "LIVE" or live_enabled:
        return False
    return _env_truthy("ALLOW_OFFHOURS_TESTING") or _env_truthy(
        "SESSION_ALLOW_OUT_OF_HOURS"
    )


def _now_ist() -> datetime:
    return datetime.now(IST)


def get_market_state() -> MarketState:
    """Args: none; Returns: current market state; Raises: none."""
    if _override_enabled():
        return MarketState.OPEN

    now = _now_ist()
    if now.weekday() >= 5:
        return MarketState.CLOSED

    current = now.time()
    preopen_start = dtime(9, 0)
    if preopen_start <= current < MARKET_OPEN:
        return MarketState.PREOPEN
    if MARKET_OPEN <= current <= MARKET_CLOSE:
        return MarketState.OPEN
    if MARKET_CLOSE < current <= dtime(16, 0):
        return MarketState.POSTMARKET
    return MarketState.CLOSED


def is_market_open_session(allow_override: bool = True) -> bool:
    if allow_override and _override_enabled():
        return True
    now_dt = _now_ist()
    if now_dt.weekday() >= 5:
        return False
    now = now_dt.time()
    return MARKET_OPEN <= now <= MARKET_CLOSE


def is_safe_entry_window(allow_override: bool = True) -> bool:
    if allow_override and _override_enabled():
        return True
    now_dt = _now_ist()
    if now_dt.weekday() >= 5:
        return False
    now = now_dt.time()
    return SAFE_START <= now <= SAFE_END


def is_market_hours(allow_override: bool = True) -> bool:
    """Backward-compatible alias for safe new-entry window."""
    return is_safe_entry_window(allow_override=allow_override)


def is_market_open() -> bool:
    """Args: none; Returns: bool; Raises: none."""
    return get_market_state() == MarketState.OPEN


def get_market_session_state(now: datetime | None = None) -> str:
    """Return one of "open" | "closed" | "preopen" | "unknown" for the given
    instant (defaults to current IST time)."""
    try:
        if now is None:
            current = _now_ist()
        else:
            current = now.astimezone(IST) if now.tzinfo else now.replace(tzinfo=IST)
    except Exception:
        return "unknown"

    if _override_enabled():
        return "open"

    if current.weekday() >= 5:
        return "closed"

    t = current.time()
    preopen_start = dtime(9, 0)
    if preopen_start <= t < MARKET_OPEN:
        return "preopen"
    if MARKET_OPEN <= t <= MARKET_CLOSE:
        return "open"
    return "closed"


def is_market_open_now() -> bool:
    """Args: none; Returns: True iff session state is open. Raises: none."""
    return get_market_session_state() == "open"


def _env_float(name: str, default: float) -> float:
    """Args: env var name, default; Returns: float; Raises: none."""
    raw = os.getenv(name)
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def is_nifty_index_symbol(symbol: str) -> bool:
    """Args: symbol; Returns: True for NIFTY spot/index variants; Raises: none."""
    upper = (symbol or "").upper().strip()
    if not upper:
        return False
    if upper in {
        "NIFTY",
        "NSE:NIFTY",
        "NSE:NIFTY 50",
        "NIFTY 50",
        "NIFTY50",
        "NSE:NIFTY50",
        "NSE:NIFTYBANK",
        "NIFTY BANK",
        "BANKNIFTY",
        "NSE:BANKNIFTY",
    }:
        return True
    if upper.endswith("CE") or upper.endswith("PE") or upper.endswith("FUT"):
        return False
    if "NIFTY" in upper and ":" in upper and not any(
        upper.endswith(suf) for suf in ("CE", "PE", "FUT")
    ):
        return True
    return False


def is_nifty_future_symbol(symbol: str) -> bool:
    """Args: symbol; Returns: True if symbol is a NIFTY/BANKNIFTY future; Raises: none."""
    upper = (symbol or "").upper().strip()
    return upper.endswith("FUT")


def is_nifty_option_symbol(symbol: str) -> bool:
    """Args: symbol; Returns: True if symbol is an NFO option (CE/PE); Raises: none."""
    upper = (symbol or "").upper().strip()
    return upper.endswith("CE") or upper.endswith("PE")


def stale_threshold_for_symbol(symbol: str, market_open: bool) -> float:
    """Return the canonical staleness threshold (seconds) for a symbol.

    Index/Future spots: 120s when market open, 3600s otherwise.
    Options:           900s when market open, 3600s otherwise.
    Anything else:     60s when market open, 3600s otherwise.

    Configurable via MDM_*_LTP_STALE_SECONDS / MDM_OFFMARKET_*_LTP_STALE_SECONDS env vars.
    """
    if is_nifty_index_symbol(symbol):
        if market_open:
            return _env_float("MDM_INDEX_LTP_STALE_SECONDS", 120.0)
        return _env_float("MDM_OFFMARKET_INDEX_LTP_STALE_SECONDS", 3600.0)
    if is_nifty_future_symbol(symbol):
        if market_open:
            return _env_float("MDM_FUTURE_LTP_STALE_SECONDS", 120.0)
        return _env_float("MDM_OFFMARKET_FUTURE_LTP_STALE_SECONDS", 3600.0)
    if is_nifty_option_symbol(symbol):
        if market_open:
            return _env_float("MDM_OPTION_LTP_STALE_SECONDS", 900.0)
        return _env_float("MDM_OFFMARKET_OPTION_LTP_STALE_SECONDS", 3600.0)
    if market_open:
        return _env_float("MDM_GENERIC_LTP_STALE_SECONDS", 60.0)
    return _env_float("MDM_OFFMARKET_GENERIC_LTP_STALE_SECONDS", 3600.0)


def get_time_status() -> Tuple[bool, str]:
    """
    Get detailed time status for logging.
    
    Returns:
        Tuple of (is_allowed, reason_string)
    """
    if _override_enabled():
        return True, "Override enabled (SESSION_ALLOW_OUT_OF_HOURS=true)"

    now_dt = _now_ist()
    if now_dt.weekday() >= 5:
        return False, f"Weekend closure ({now_dt.strftime('%A')})"

    now = now_dt.time()

    if now < MARKET_OPEN:
        return False, f"Pre-market ({now.strftime('%H:%M')} < {MARKET_OPEN.strftime('%H:%M')})"
    
    if MARKET_OPEN <= now < SAFE_START:
        return False, f"Opening volatility buffer (Wait until {SAFE_START.strftime('%H:%M')})"
    
    if SAFE_START <= now <= SAFE_END:
        return True, "Within safe entry window"

    if SAFE_END < now <= MARKET_CLOSE:
        return False, f"EOD safety cutoff (No new entries after {SAFE_END.strftime('%H:%M')})"

    return False, f"Market closed after {MARKET_CLOSE.strftime('%H:%M')}"


def get_current_ist_time() -> datetime:
    """Get current time in IST."""
    return _now_ist()


def format_time_for_log() -> str:
    """Get formatted time string for logging."""
    return _now_ist().strftime("%H:%M:%S IST")


# Cached version for high-frequency calls (1 second cache)
@lru_cache(maxsize=128)
def _cached_market_hours_check(minute_key: int, override_flag: str) -> bool:
    """Internal cached check - cache key changes every minute + override state."""
    return is_market_hours(allow_override=True)


@lru_cache(maxsize=128)
def _cached_time_status_check(minute_key: int, override_flag: str) -> tuple[bool, str]:
    return get_time_status()


def get_time_status_cached() -> tuple[bool, str]:
    now = datetime.now(IST)
    minute_key = now.hour * 60 + now.minute
    override_flag = "|".join(
        [
            os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower(),
            os.getenv("ALLOW_OFFHOURS_TESTING", "").lower(),
            os.getenv("EXECUTION_MODE", "").lower(),
            os.getenv("ENABLE_LIVE", "").lower(),
            os.getenv("ENABLE_LIVE_TRADING", "").lower(),
            os.getenv("SAFE_START_TIME", "").lower(),
            os.getenv("SAFE_END_TIME", "").lower(),
            os.getenv("MARKET_OPEN_TIME", "").lower(),
            os.getenv("MARKET_CLOSE_TIME", "").lower(),
        ]
    )
    return _cached_time_status_check(minute_key, override_flag)


def is_market_hours_cached() -> bool:
    """
    Cached version of is_market_hours for high-frequency tick processing.
    Cache refreshes every minute and when SESSION_ALLOW_OUT_OF_HOURS changes.
    """
    now = datetime.now(IST)
    minute_key = now.hour * 60 + now.minute
    override_flag = "|".join(
        [
            os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower(),
            os.getenv("ALLOW_OFFHOURS_TESTING", "").lower(),
            os.getenv("EXECUTION_MODE", "").lower(),
            os.getenv("ENABLE_LIVE", "").lower(),
            os.getenv("ENABLE_LIVE_TRADING", "").lower(),
            os.getenv("SAFE_START_TIME", "").lower(),
            os.getenv("SAFE_END_TIME", "").lower(),
            os.getenv("MARKET_OPEN_TIME", "").lower(),
            os.getenv("MARKET_CLOSE_TIME", "").lower(),
        ]
    )
    return _cached_market_hours_check(minute_key, override_flag)


__all__ = [
    "is_market_hours",
    "is_market_open_session",
    "is_safe_entry_window",
    "is_market_open",
    "is_market_open_now",
    "is_market_hours_cached",
    "get_time_status",
    "get_time_status_cached",
    "get_current_ist_time",
    "format_time_for_log",
    "get_market_session_state",
    "MarketState",
    "get_market_state",
    "IST",
    "MARKET_OPEN",
    "SAFE_START",
    "SAFE_END",
    "MARKET_CLOSE",
    "allow_offhours_testing_safe",
    "stale_threshold_for_symbol",
    "is_nifty_index_symbol",
    "is_nifty_future_symbol",
    "is_nifty_option_symbol",
]
