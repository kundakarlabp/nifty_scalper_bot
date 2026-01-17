"""Trading session gate helpers shared across components.

MODIFIED: Added comprehensive logging to diagnose trading blocks.
"""

from __future__ import annotations

import logging
from datetime import datetime, time, timedelta
from typing import Mapping
from zoneinfo import ZoneInfo

from nifty_scalper_bot.utils.env import get_bool

IST = ZoneInfo("Asia/Kolkata")
MARKET_OPEN = time(9, 15)
MARKET_CLOSE = time(15, 30)

# Setup module logger
_LOGGER = logging.getLogger("nifty_scalper_bot.risk.session_gate")


def _normalize_time(value: time) -> time:
    if value.tzinfo is not None:
        return value
    return value.replace(tzinfo=IST)


def _is_market_open(now: datetime, *, start: time, end: time) -> bool:
    ist_now = now.astimezone(IST)
    start_time = _normalize_time(start)
    end_time = _normalize_time(end)
    window_start = ist_now.replace(
        hour=start_time.hour,
        minute=start_time.minute,
        second=start_time.second,
        microsecond=0,
    )
    window_end = ist_now.replace(
        hour=end_time.hour,
        minute=end_time.minute,
        second=end_time.second,
        microsecond=0,
    )
    if window_end <= window_start:
        window_end = window_end + timedelta(days=1)
    return window_start <= ist_now <= window_end


def build_session_guard(
    now: datetime | None = None,
    *,
    override: bool | None = None,
    market_open: time | None = None,
    market_close: time | None = None,
) -> dict[str, object]:
    """Return deterministic session guard payload based on time and overrides."""

    current = now or datetime.now(IST)
    if current.tzinfo is None:
        current = current.replace(tzinfo=IST)
    open_time = market_open or MARKET_OPEN
    close_time = market_close or MARKET_CLOSE
    market_open_flag = _is_market_open(current, start=open_time, end=close_time)
    override_flag = (
        bool(override)
        if override is not None
        else get_bool(
            "SESSION_ALLOW_OUT_OF_HOURS", "ALLOW_OFFHOURS_TESTING", default=False
        )
    )
    session_valid = bool(market_open_flag or override_flag)
    reasons: list[str] = [] if session_valid else ["Outside trading window"]
    
    # Log session state for debugging
    _LOGGER.debug(
        f"Session guard: market_open={market_open_flag}, override={override_flag}, "
        f"session_valid={session_valid}, time={current.strftime('%H:%M:%S')} IST"
    )
    
    return {
        "market_open": market_open_flag,
        "override_out_of_hours": override_flag,
        "session_valid": session_valid,
        "rate_limits_ok": True,
        "risk_green": True,
        "reasons": reasons,
        "timestamp": current.astimezone(IST).isoformat(),
    }


def can_trade(
    guard: Mapping[str, object],
    *,
    enable_live: bool,
    shadow_mode: bool,
) -> bool:
    """Return ``True`` when trading is permitted under the provided guard.
    
    ENHANCED: Added logging to diagnose trading blocks.
    """

    # Check 1: Session validity (market hours)
    if not guard.get("session_valid", False):
        _LOGGER.warning(
            "🚫 TRADE BLOCKED: Session not valid (outside market hours 9:15-15:30 IST). "
            "Set SESSION_ALLOW_OUT_OF_HOURS=true for testing."
        )
        return False
    
    # Check 2: Live trading enabled
    if not enable_live:
        _LOGGER.warning(
            "🚫 TRADE BLOCKED: ENABLE_LIVE=false. "
            "Set ENABLE_LIVE=true in Railway environment to enable live orders."
        )
        return False
    
    # Check 3: Shadow mode
    if shadow_mode:
        _LOGGER.warning(
            "🚫 TRADE BLOCKED: Shadow mode is active. "
            "Disable shadow mode for live trading."
        )
        return False
    
    # Check 4: Rate limits
    if not guard.get("rate_limits_ok", False):
        _LOGGER.warning("🚫 TRADE BLOCKED: Rate limits exceeded. Wait for cooldown.")
        return False
    
    # Check 5: Risk manager
    if not guard.get("risk_green", False):
        _LOGGER.warning(
            "🚫 TRADE BLOCKED: Risk manager red. "
            "Check daily loss limits, max positions, etc."
        )
        return False
    
    # Check 6: Broker session
    broker_valid = guard.get("broker_session_valid")
    if broker_valid is not None and not bool(broker_valid):
        _LOGGER.warning(
            "🚫 TRADE BLOCKED: Broker session invalid. "
            "Refresh Zerodha access token."
        )
        return False
    
    # All checks passed
    _LOGGER.debug("✅ All trading gates PASSED - order can be placed")
    return True


__all__ = ["IST", "MARKET_OPEN", "MARKET_CLOSE", "build_session_guard", "can_trade"]
