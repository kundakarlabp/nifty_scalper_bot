from nifty_scalper_bot.utils.ist_clock import IST_NAME, minute, timestamp


def test_naive_time_is_ist():
    ts = timestamp('2026-07-06 09:15:30')
    assert str(ts.tz) == IST_NAME
    assert ts.hour == 9
    assert ts.minute == 15


def test_z_time_converts_to_ist():
    ts = minute('2026-07-06T03:45:59Z')
    assert str(ts.tz) == IST_NAME
    assert ts.isoformat() == '2026-07-06T09:15:00+05:30'
