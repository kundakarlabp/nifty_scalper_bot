from datetime import datetime
from zoneinfo import ZoneInfo

from src.risk.greeks import next_weekly_expiry_ist


def test_expiry_monday_to_thursday():
    now = datetime(2024, 7, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Monday
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == datetime(2024, 7, 4).date()
    assert expiry.hour == 15 and expiry.minute == 30


def test_expiry_thursday_before_time():
    now = datetime(2024, 7, 4, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == now.date()
    assert expiry.hour == 15 and expiry.minute == 30


def test_expiry_thursday_after_time():
    now = datetime(2024, 7, 4, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == datetime(2024, 7, 11).date()
    assert expiry.hour == 15 and expiry.minute == 30
