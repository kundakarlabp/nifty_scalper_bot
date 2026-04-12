# src/nifty_scalper_bot/utils/smart_symbol.py
#
# ⚠️  LEGACY CE/PE SYMBOL-BUILDING FUNCTIONS REMOVED.
#
# All option contract selection now goes through:
#   core/contract_selector.py  → get_atm_contracts(kite, underlying_price)
#   core/instrument_manager.py → InstrumentManager.get_token(symbol)
#   core/hydration.py          → hydrate(kite, token)
#
# Keeping only the pure date/calendar utilities that risk and strategy
# modules legitimately depend on.

from datetime import datetime, timedelta, date
from zoneinfo import ZoneInfo
import logging

logger = logging.getLogger(__name__)

IST = ZoneInfo("Asia/Kolkata")
WEEKLY_EXPIRY_WEEKDAY = 1  # Tuesday (0=Mon, 1=Tue, …)

# NSE market holidays — update annually
NSE_HOLIDAYS = {
    date(2026, 1, 26),   # Republic Day
    date(2026, 3, 20),   # Id-ul-Fitr
    date(2026, 4, 14),   # Dr. Baba Saheb Ambedkar Jayanti
    date(2026, 5, 1),    # Maharashtra Day
    date(2026, 10, 2),   # Gandhi Jayanti
    date(2026, 12, 25),  # Christmas
}


def now_ist() -> datetime:
    """Return current datetime in IST."""
    return datetime.now(IST)


def get_actual_expiry_date(start_date: date, target_weekday: int) -> date:
    """Return the nearest *target_weekday* on or after *start_date*, adjusted
    backwards past NSE holidays and weekends.

    Args:
        start_date: Reference date.
        target_weekday: 0=Mon … 6=Sun.

    Returns:
        date: Adjusted expiry date.

    Raises:
        None.
    """
    days_ahead = target_weekday - start_date.weekday()
    if days_ahead < 0:
        days_ahead += 7
    expiry = start_date + timedelta(days=days_ahead)
    while expiry in NSE_HOLIDAYS or expiry.weekday() >= 5:
        expiry -= timedelta(days=1)
    return expiry


def next_weekday(start_date: date, target_weekday: int) -> date:
    """Holiday-adjusted *target_weekday* on or after *start_date*.

    Args:
        start_date: Reference date.
        target_weekday: 0=Mon … 6=Sun.

    Returns:
        date: Adjusted date.

    Raises:
        None.
    """
    return get_actual_expiry_date(start_date, target_weekday)
