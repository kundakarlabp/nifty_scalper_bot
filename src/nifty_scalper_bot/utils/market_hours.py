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
MARKET_OPEN = dtime(9, 15)      # NSE opens
SAFE_START = dtime(9, 20)       # Avoid opening volatility (was 9:30, now 9:20)
SAFE_END = dtime(15, 15)        # Stop new entries 15 mins before close
MARKET_CLOSE = dtime(15, 30)    # NSE closes


class MarketState(Enum):
    """Market session state classification."""

    CLOSED = 'closed'
    PREOPEN = 'preopen'
    OPEN = 'open'
    POSTMARKET = 'postmarket'


def get_market_state() -> MarketState:
    """Args: none; Returns: current market state; Raises: none."""
    override = os.getenv('SESSION_ALLOW_OUT_OF_HOURS', '').lower()
    if override == 'true':
        return MarketState.OPEN

    now = datetime.now(IST)
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


def is_market_hours(allow_override: bool = True) -> bool:
    """
    Check if current time is within safe trading hours.
    
    Args:
        allow_override: If True, checks SESSION_ALLOW_OUT_OF_HOURS env var
        
    Returns:
        True if trading is allowed, False otherwise
    """
    # Check environment override for testing
    if allow_override:
        override = os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower()
        if override == "true":
            return True
    
    now = datetime.now(IST).time()
    return SAFE_START <= now <= SAFE_END


def is_market_open() -> bool:
    """
    Check if market is open (broader check - 9:15 to 15:30).
    Useful for data collection even when not trading.
    """
    return get_market_state() == MarketState.OPEN


def get_time_status() -> Tuple[bool, str]:
    """
    Get detailed time status for logging.
    
    Returns:
        Tuple of (is_allowed, reason_string)
    """
    # Check override first
    override = os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower()
    if override == "true":
        return True, "Override enabled (SESSION_ALLOW_OUT_OF_HOURS=true)"
    
    now = datetime.now(IST).time()
    
    if now < MARKET_OPEN:
        return False, f"Pre-market ({now.strftime('%H:%M')} < {MARKET_OPEN.strftime('%H:%M')})"
    
    if MARKET_OPEN <= now < SAFE_START:
        return False, f"Opening volatility buffer (Wait until {SAFE_START.strftime('%H:%M')})"
    
    if SAFE_START <= now <= SAFE_END:
        return True, "Within safe trading window"
    
    if SAFE_END < now <= MARKET_CLOSE:
        return False, f"EOD safety cutoff (No trades after {SAFE_END.strftime('%H:%M')})"
    
    return False, f"Market closed ({now.strftime('%H:%M')} > {MARKET_CLOSE.strftime('%H:%M')})"


def get_current_ist_time() -> datetime:
    """Get current time in IST."""
    return datetime.now(IST)


def format_time_for_log() -> str:
    """Get formatted time string for logging."""
    return datetime.now(IST).strftime("%H:%M:%S IST")


# Cached version for high-frequency calls (1 second cache)
@lru_cache(maxsize=128)
def _cached_market_hours_check(minute_key: int, override_flag: str) -> bool:
    """Internal cached check - cache key changes every minute and override state."""
    return is_market_hours(allow_override=True)


def is_market_hours_cached() -> bool:
    """
    Cached version of is_market_hours for high-frequency tick processing.
    Cache refreshes every minute.
    """
    now = datetime.now(IST)
    minute_key = now.hour * 60 + now.minute
    override_flag = os.getenv("SESSION_ALLOW_OUT_OF_HOURS", "").lower()
    return _cached_market_hours_check(minute_key, override_flag)


__all__ = [
    "is_market_hours",
    "is_market_open", 
    "is_market_hours_cached",
    "get_time_status",
    "get_current_ist_time",
    "format_time_for_log",
    "MarketState",
    "get_market_state",
    "IST",
    "MARKET_OPEN",
    "SAFE_START", 
    "SAFE_END",
    "MARKET_CLOSE",
]
