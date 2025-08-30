from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils.events import load_calendar
from src.utils import events as events_module


def test_next_event(monkeypatch):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2025, 9, 1, tzinfo=tz)

    monkeypatch.setattr(events_module, "datetime", FixedDateTime)
    cal = load_calendar("config/events.yaml")
    now = datetime(2025, 9, 1, 8, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    ev = cal.next_event(now)
    assert ev is not None
    assert ev.name == "RBI Policy"
    assert ev.guard_start() == datetime(2025, 9, 1, 9, 50, tzinfo=ZoneInfo("Asia/Kolkata"))
