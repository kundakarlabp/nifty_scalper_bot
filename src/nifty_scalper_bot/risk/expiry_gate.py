"""Expiry-day theta gate for new option-buy entries.

On NIFTY weekly expiry day (Tuesday), ATM option premiums decay rapidly in
the afternoon; long-option scalps need a much larger move just to break
even. This gate blocks NEW option-buy candidates after a cutoff time on
expiry day. It never touches exits or option-selling (decay) strategies.

Env overrides:
- EXPIRY_THETA_GATE_ENABLED (default true)
- EXPIRY_ENTRY_CUTOFF_IST   (default "13:30", HH:MM, IST)
- EXPIRY_WEEKDAY            (default 1 = Tuesday, Monday=0)
- ALLOW_EXPIRY_DAY_AFTER_CUTOFF (default false)
"""

from __future__ import annotations

import os
from datetime import datetime, time as dtime, timedelta, timezone

from nifty_scalper_bot.config.env_utils import parse_bool_env, parse_int_env

IST = timezone(timedelta(hours=5, minutes=30))


def _cutoff_time() -> dtime:
    raw = str(os.getenv("EXPIRY_ENTRY_CUTOFF_IST") or "13:30").strip()
    try:
        hh, mm = raw.split(":", 1)
        return dtime(hour=max(0, min(23, int(hh))), minute=max(0, min(59, int(mm))))
    except (ValueError, AttributeError):
        return dtime(hour=13, minute=30)


def expiry_theta_block(now: datetime | None = None) -> tuple[bool, str]:
    """Args: optional now (tz-aware). Returns: (blocked, reason). Raises: none.

    blocked=True means new option-buy entries should be rejected because it
    is expiry day at/after the configured IST cutoff.
    """
    if not parse_bool_env(os.getenv("EXPIRY_THETA_GATE_ENABLED"), True):
        return False, "gate_disabled"
    if parse_bool_env(os.getenv("ALLOW_EXPIRY_DAY_AFTER_CUTOFF"), False):
        return False, "expiry_after_cutoff_allowed"
    current = now.astimezone(IST) if now else datetime.now(IST)
    expiry_weekday = parse_int_env(os.getenv("EXPIRY_WEEKDAY"), 1)
    if current.weekday() != expiry_weekday:
        return False, "not_expiry_day"
    cutoff = _cutoff_time()
    if current.time() < cutoff:
        return False, "before_cutoff"
    return True, f"expiry_day_after_{cutoff.strftime('%H:%M')}_ist"


def _env_hhmm(name: str, default: dtime) -> dtime:
    raw = str(os.getenv(name) or "").strip()
    try:
        hh, mm = raw.split(":", 1)
        hour = int(hh)
        minute = int(mm)
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return default
        return dtime(hour=hour, minute=minute)
    except (ValueError, AttributeError):
        return default


def midday_pause_block(now: datetime | None = None) -> tuple[bool, str]:
    """Args: optional now (tz-aware). Returns: (blocked, reason). Raises: none.

    Blocks NEW option-buy entries during the low-volatility midday window
    (default 11:30-13:15 IST) where chop feeds transaction costs. Exits
    untouched. Env: MIDDAY_PAUSE_ENABLED (false), MIDDAY_PAUSE_START (11:30),
    MIDDAY_PAUSE_END (13:15).
    """
    if not parse_bool_env(os.getenv("MIDDAY_PAUSE_ENABLED"), False):
        return False, "pause_disabled"
    current = (now.astimezone(IST) if now else datetime.now(IST)).time()
    start = _env_hhmm("MIDDAY_PAUSE_START", dtime(11, 30))
    end = _env_hhmm("MIDDAY_PAUSE_END", dtime(13, 15))
    if start <= current < end:
        return True, f"midday_pause_{start.strftime('%H:%M')}-{end.strftime('%H:%M')}_ist"
    return False, "outside_pause"
