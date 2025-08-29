from __future__ import annotations

from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from src.config import settings

TZ_NAME = getattr(settings, "tz", "Asia/Kolkata")
TZ = ZoneInfo(TZ_NAME)

def now_ist() -> datetime:
    """Return current time in configured timezone (aware)."""
    return datetime.now(TZ)

def floor_to_minute(ts: datetime, tz: ZoneInfo | None = None) -> datetime:
    """Floor a datetime to the minute in the given timezone.

    If ``ts`` is naive it is assumed to be in the target timezone.
    """
    tz = tz or TZ
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=tz)
    return ts.astimezone(tz).replace(second=0, microsecond=0)
