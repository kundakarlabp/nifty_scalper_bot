from datetime import datetime

from src.utils.events import load_calendar
from src.utils import events as events_module


def test_events_parsing(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 9, 1, tzinfo=tz)

    monkeypatch.setattr(events_module, "datetime", FixedDateTime)
    cal = load_calendar("config/events.yaml")
    assert cal.tz.key == "Asia/Kolkata"
    assert cal.defaults.guard_before_min == 5
    assert len(cal.events) == 5
    names = [e.name for e in cal.events]
    assert "Expiry Last 35m" in names
