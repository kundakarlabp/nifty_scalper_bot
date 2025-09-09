from __future__ import annotations

from datetime import datetime
import logging
from zoneinfo import ZoneInfo

DEFAULT_TZ = "Asia/Kolkata"
_log = logging.getLogger(__name__)
_TZ: ZoneInfo | None = None


def get_timezone() -> ZoneInfo:
    """Return configured timezone or fall back to DEFAULT_TZ with warning."""
    global _TZ
    if _TZ is not None:
        return _TZ
    tz_name = DEFAULT_TZ
    try:
        from src.config import settings  # deferred to avoid heavy import at module load
        tz_name = getattr(settings, "tz", DEFAULT_TZ)
    except Exception:
        tz_name = DEFAULT_TZ
    try:
        _TZ = ZoneInfo(tz_name)
    except Exception:
        _log.warning("Falling back to default timezone: %s", DEFAULT_TZ)
        _TZ = ZoneInfo(DEFAULT_TZ)
    return _TZ


TZ = get_timezone()


def now_ist() -> datetime:
    """Return current time in configured timezone."""
    return datetime.now(TZ)


def floor_to_minute(ts: datetime, tz: ZoneInfo | None = None) -> datetime:
    """Floor ``ts`` to the nearest minute in the given timezone."""
    tz = tz or TZ
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz)
    ts = ts.astimezone(tz)
    return ts.replace(second=0, microsecond=0)

__all__ = ["now_ist", "floor_to_minute", "TZ", "get_timezone"]
