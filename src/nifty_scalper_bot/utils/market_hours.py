"""Canonical NSE market-session and safe-entry window utilities."""

from __future__ import annotations

import os
from datetime import date, datetime, time as dtime
from enum import Enum
from functools import lru_cache
from typing import Literal, Tuple
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.logging import get_logger
from nifty_scalper_bot.utils.nse_calendar import holiday_name as nse_holiday_name

LOGGER = get_logger(__name__)
IST = ZoneInfo("Asia/Kolkata")


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


MARKET_OPEN = _env_time("MARKET_OPEN_TIME", dtime(9, 15))
SAFE_START = _env_time("SAFE_START_TIME", dtime(9, 20))
SAFE_END = _env_time("SAFE_END_TIME", dtime(15, 25))
MARKET_CLOSE = _env_time("MARKET_CLOSE_TIME", dtime(15, 30))


class MarketState(Enum):
    """Market session state classification."""

    CLOSED = "closed"
    PREOPEN = "preopen"
    OPEN = "open"
    POSTMARKET = "postmarket"


def allow_offhours_testing_safe() -> bool:
    """Return whether off-hours testing is allowed without live-order risk."""
    execution_mode = os.getenv("EXECUTION_MODE", "SHADOW").strip().upper()
    live_enabled = _env_truthy("ENABLE_LIVE") or _env_truthy("ENABLE_LIVE_TRADING")
    if execution_mode == "LIVE" or live_enabled:
        return False
    return _env_truthy("ALLOW_OFFHOURS_TESTING") or _env_truthy(
        "SESSION_ALLOW_OUT_OF_HOURS"
    )


def _override_enabled() -> bool:
    return allow_offhours_testing_safe()


def _now_ist() -> datetime:
    return datetime.now(IST)


def _coerce_ist(value: datetime | None) -> datetime:
    if value is None:
        return _now_ist()
    return value.astimezone(IST) if value.tzinfo else value.replace(tzinfo=IST)


def exchange_holiday_name(value: datetime | date | None = None) -> str | None:
    """Return the NSE F&O holiday name for the supplied/current IST date."""
    if value is None:
        day = _now_ist().date()
    elif isinstance(value, datetime):
        day = _coerce_ist(value).date()
    else:
        day = value
    return nse_holiday_name(day)


def is_exchange_holiday(value: datetime | date | None = None) -> bool:
    """Return whether the supplied/current IST date is an NSE F&O holiday."""
    return exchange_holiday_name(value) is not None


def _is_non_trading_day(now: datetime) -> bool:
    return now.weekday() >= 5 or is_exchange_holiday(now)


def get_market_state() -> MarketState:
    """Return the current canonical NSE session state."""
    if _override_enabled():
        return MarketState.OPEN
    now = _now_ist()
    if _is_non_trading_day(now):
        return MarketState.CLOSED
    current = now.time()
    if dtime(9, 0) <= current < MARKET_OPEN:
        return MarketState.PREOPEN
    if MARKET_OPEN <= current <= MARKET_CLOSE:
        return MarketState.OPEN
    if MARKET_CLOSE < current <= dtime(16, 0):
        return MarketState.POSTMARKET
    return MarketState.CLOSED


def get_runtime_market_mode() -> Literal[
    "PRE_MARKET", "OPEN", "POST_MARKET", "HOLIDAY", "UNKNOWN"
]:
    """Return the operator-facing runtime market mode."""
    try:
        if _override_enabled():
            return "OPEN"
        now = _now_ist()
        if _is_non_trading_day(now):
            return "HOLIDAY"
        state = get_market_state()
        if state == MarketState.OPEN:
            return "OPEN"
        if state == MarketState.PREOPEN:
            return "PRE_MARKET"
        if state == MarketState.POSTMARKET:
            return "POST_MARKET"
        if now.time() < MARKET_OPEN:
            return "PRE_MARKET"
        if now.time() > MARKET_CLOSE:
            return "POST_MARKET"
        return "UNKNOWN"
    except Exception:
        return "UNKNOWN"


def post_market_quiet_mode_enabled() -> bool:
    return os.getenv("POST_MARKET_QUIET_MODE", "true").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def post_market_basket_refresh_seconds() -> float:
    try:
        return max(
            60.0,
            float(os.getenv("POST_MARKET_BASKET_REFRESH_SECONDS", "900") or 900),
        )
    except (TypeError, ValueError):
        return 900.0


def post_market_market_data_summary_seconds() -> float:
    try:
        return max(
            60.0,
            float(
                os.getenv("POST_MARKET_MARKET_DATA_SUMMARY_SECONDS", "300") or 300
            ),
        )
    except (TypeError, ValueError):
        return 300.0


def post_market_suppress_candle_gap_warnings() -> bool:
    return os.getenv(
        "POST_MARKET_SUPPRESS_CANDLE_GAP_WARNINGS", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}


def is_market_open_session(allow_override: bool = True) -> bool:
    if allow_override and _override_enabled():
        return True
    now = _now_ist()
    if _is_non_trading_day(now):
        return False
    return MARKET_OPEN <= now.time() <= MARKET_CLOSE


def is_safe_entry_window(allow_override: bool = True) -> bool:
    if allow_override and _override_enabled():
        return True
    now = _now_ist()
    if _is_non_trading_day(now):
        return False
    return SAFE_START <= now.time() <= SAFE_END


def is_market_hours(allow_override: bool = True) -> bool:
    """Backward-compatible alias for the safe new-entry window."""
    return is_safe_entry_window(allow_override=allow_override)


def is_market_open() -> bool:
    return get_market_state() == MarketState.OPEN


def get_market_session_state(now: datetime | None = None) -> str:
    """Return ``open``, ``closed``, ``preopen`` or ``unknown`` for an instant."""
    try:
        current = _coerce_ist(now)
    except Exception:
        return "unknown"
    if _override_enabled():
        return "open"
    if _is_non_trading_day(current):
        return "closed"
    current_time = current.time()
    if dtime(9, 0) <= current_time < MARKET_OPEN:
        return "preopen"
    if MARKET_OPEN <= current_time <= MARKET_CLOSE:
        return "open"
    return "closed"


def is_market_open_now() -> bool:
    return get_market_session_state() == "open"


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if not raw:
        return float(default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return float(default)


def is_nifty_index_symbol(symbol: str) -> bool:
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
    if upper.endswith(("CE", "PE", "FUT")):
        return False
    return "NIFTY" in upper and ":" in upper


def is_nifty_future_symbol(symbol: str) -> bool:
    return (symbol or "").upper().strip().endswith("FUT")


def is_nifty_option_symbol(symbol: str) -> bool:
    upper = (symbol or "").upper().strip()
    return upper.endswith("CE") or upper.endswith("PE")


def stale_threshold_for_symbol(symbol: str, market_open: bool) -> float:
    """Return the canonical staleness threshold in seconds for a symbol."""
    if is_nifty_index_symbol(symbol):
        return _env_float(
            "MDM_INDEX_LTP_STALE_SECONDS" if market_open else "MDM_OFFMARKET_INDEX_LTP_STALE_SECONDS",
            120.0 if market_open else 3600.0,
        )
    if is_nifty_future_symbol(symbol):
        return _env_float(
            "MDM_FUTURE_LTP_STALE_SECONDS" if market_open else "MDM_OFFMARKET_FUTURE_LTP_STALE_SECONDS",
            120.0 if market_open else 3600.0,
        )
    if is_nifty_option_symbol(symbol):
        return _env_float(
            "MDM_OPTION_LTP_STALE_SECONDS" if market_open else "MDM_OFFMARKET_OPTION_LTP_STALE_SECONDS",
            900.0 if market_open else 3600.0,
        )
    return _env_float(
        "MDM_GENERIC_LTP_STALE_SECONDS" if market_open else "MDM_OFFMARKET_GENERIC_LTP_STALE_SECONDS",
        60.0 if market_open else 3600.0,
    )


def get_time_status() -> Tuple[bool, str]:
    """Return whether new entries are time-allowed and a diagnostic reason."""
    if _override_enabled():
        return True, "Override enabled (SESSION_ALLOW_OUT_OF_HOURS=true)"
    now = _now_ist()
    if now.weekday() >= 5:
        return False, f"Weekend closure ({now.strftime('%A')})"
    holiday = exchange_holiday_name(now)
    if holiday:
        return False, f"Exchange holiday ({holiday})"
    current = now.time()
    if current < MARKET_OPEN:
        return False, (
            f"Pre-market ({current.strftime('%H:%M')} < "
            f"{MARKET_OPEN.strftime('%H:%M')})"
        )
    if MARKET_OPEN <= current < SAFE_START:
        return False, (
            f"Opening volatility buffer (Wait until {SAFE_START.strftime('%H:%M')})"
        )
    if SAFE_START <= current <= SAFE_END:
        return True, "Within safe entry window"
    if SAFE_END < current <= MARKET_CLOSE:
        return False, (
            f"EOD safety cutoff (No new entries after {SAFE_END.strftime('%H:%M')})"
        )
    return False, f"Market closed after {MARKET_CLOSE.strftime('%H:%M')}"


def get_current_ist_time() -> datetime:
    return _now_ist()


def format_time_for_log() -> str:
    return _now_ist().strftime("%H:%M:%S IST")


def _cache_minute_key(now: datetime) -> int:
    """Return a date-aware key so yesterday's state is never reused."""
    return now.date().toordinal() * 1440 + now.hour * 60 + now.minute


def _runtime_cache_flags() -> str:
    return "|".join(
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
            os.getenv("NSE_ADDITIONAL_HOLIDAYS", "").lower(),
            os.getenv("NSE_REMOVED_HOLIDAYS", "").lower(),
        ]
    )


@lru_cache(maxsize=128)
def _cached_market_hours_check(minute_key: int, override_flag: str) -> bool:
    del minute_key, override_flag
    return is_market_hours(allow_override=True)


@lru_cache(maxsize=128)
def _cached_time_status_check(minute_key: int, override_flag: str) -> tuple[bool, str]:
    del minute_key, override_flag
    return get_time_status()


def get_time_status_cached() -> tuple[bool, str]:
    now = _now_ist()
    return _cached_time_status_check(_cache_minute_key(now), _runtime_cache_flags())


def is_market_hours_cached() -> bool:
    """Return cached safe-window state, refreshed each IST calendar minute."""
    now = _now_ist()
    return _cached_market_hours_check(_cache_minute_key(now), _runtime_cache_flags())


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
    "get_runtime_market_mode",
    "MarketState",
    "get_market_state",
    "exchange_holiday_name",
    "is_exchange_holiday",
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
    "post_market_quiet_mode_enabled",
    "post_market_basket_refresh_seconds",
    "post_market_market_data_summary_seconds",
    "post_market_suppress_candle_gap_warnings",
]
