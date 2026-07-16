from datetime import datetime
from zoneinfo import ZoneInfo

from .virtual_clock import VirtualClock


def test_virtual_clock_progression_timezone_callbacks_and_no_real_sleep():
    clock = VirtualClock(datetime(2026, 7, 15, 8, 55, tzinfo=ZoneInfo("Asia/Kolkata")))
    seen = []
    clock.call_later(0.5, lambda: seen.append(("first", clock.monotonic())))
    clock.call_later(0.25, lambda: seen.append(("second", clock.monotonic())))
    clock.sleep(1)
    assert clock.now().tzinfo.key == "Asia/Kolkata"
    assert clock.monotonic() == 1.0
    assert [item[0] for item in seen] == ["second", "first"]
    before = clock.time()
    clock.advance(milliseconds=250)
    assert clock.time() == before + 0.25
    clock.advance_to_next_minute()
    assert clock.now().second == 0
