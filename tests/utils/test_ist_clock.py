from nifty_scalper_bot.utils.ist_clock import timestamp


def test_clock_fields():
    ts = timestamp('2026-07-06 09:15:30')
    assert ts.hour == 9
    assert ts.minute == 15
