from datetime import datetime
from zoneinfo import ZoneInfo

from src.risk.greeks import next_weekly_expiry_ist


def test_expiry_monday_to_tuesday():
    now = datetime(2024, 7, 1, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))  # Monday
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == datetime(2024, 7, 2).date()
    assert expiry.hour == 15 and expiry.minute == 30


def test_expiry_tuesday_before_time():
    now = datetime(2024, 7, 2, 10, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == now.date()
    assert expiry.hour == 15 and expiry.minute == 30


def test_expiry_tuesday_after_time():
    now = datetime(2024, 7, 2, 16, 0, tzinfo=ZoneInfo("Asia/Kolkata"))
    expiry = next_weekly_expiry_ist(now)
    assert expiry.date() == datetime(2024, 7, 9).date()
    assert expiry.hour == 15 and expiry.minute == 30
