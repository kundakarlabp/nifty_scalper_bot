from nifty_scalper_bot.utils.ist_clock import IST_NAME, timestamp


def test_naive_time_is_ist():
    ts = timestamp('2026-07-06 09:15:30')
    assert str(ts.tz) == IST_NAME
