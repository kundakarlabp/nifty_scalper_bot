from datetime import datetime
from zoneinfo import ZoneInfo

from src.utils.market_time import is_market_open, last_session_window


def test_last_session_window_weekday_evening():
    now = datetime(2025, 8, 31, 18, 30, tzinfo=ZoneInfo("Asia/Kolkata"))
    assert not is_market_open(now)
    start, end = last_session_window(now)
    assert start.hour == 9 and end.hour == 15
