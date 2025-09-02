from datetime import datetime

from freezegun import freeze_time

from src.utils.expiry import last_tuesday_of_month, next_tuesday_expiry


@freeze_time("2025-08-26 16:00:00+05:30")
def test_next_tuesday_expiry_rolls_over():
    now = datetime(2025, 8, 26, 16, 0)
    exp = next_tuesday_expiry(now)
    assert exp.date() == datetime(2025, 9, 2).date()
    assert exp.hour == 15 and exp.minute == 30


@freeze_time("2025-08-01 10:00:00+05:30")
def test_last_tuesday_of_month():
    now = datetime(2025, 8, 1, 10, 0)
    exp = last_tuesday_of_month(now)
    assert exp.date() == datetime(2025, 8, 26).date()
    assert exp.hour == 15 and exp.minute == 30
