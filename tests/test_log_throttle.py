from nifty_scalper_bot.utils.logging import LogThrottle


def test_log_throttle_basic() -> None:
    t = LogThrottle(cooldown_seconds=1.0)
    assert t.should_log('k') is True
    assert t.should_log('k') is False
