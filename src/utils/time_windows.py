from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

try:
    from src.config import settings
    _TZ_NAME = getattr(settings, "tz", "Asia/Kolkata")
except Exception:  # pragma: no cover
    _TZ_NAME = "Asia/Kolkata"

TZ = ZoneInfo(_TZ_NAME)


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

__all__ = ["now_ist", "floor_to_minute", "TZ"]
