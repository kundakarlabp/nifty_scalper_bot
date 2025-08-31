from datetime import datetime
from zoneinfo import ZoneInfo

from utils.market_time import prev_session_last_20m

IST = ZoneInfo("Asia/Kolkata")

def _w(y, m, d, h, mi):
    return datetime(y, m, d, h, mi, tzinfo=IST)

def test_prev_session_20m_after_close_friday():
    now = _w(2025, 8, 29, 18, 30)  # Fri 18:30 IST
    start, end = prev_session_last_20m(now)
    assert end.hour == 15 and end.minute == 30
    assert (end - start).total_seconds() == 20 * 60

def test_prev_session_20m_monday_preopen():
    now = _w(2025, 9, 1, 8, 30)  # Mon before open
    start, end = prev_session_last_20m(now)
    assert end.hour == 15 and end.minute == 30  # from the prior Fri
