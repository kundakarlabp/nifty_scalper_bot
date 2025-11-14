from datetime import datetime, timezone

from nifty_scalper_bot.strategies.bar_builder import OneMinuteBarBuilder


def test_one_minute_bar_builder_finalises_bar() -> None:
    builder = OneMinuteBarBuilder()
    base = datetime(2024, 1, 1, 9, 30, tzinfo=timezone.utc)

    assert builder.update(100.0, 10, base) is None
    assert builder.update(101.5, 5, base.replace(second=30)) is None

    completed = builder.update(102.0, 7, base.replace(minute=31))
    assert completed is not None
    assert completed.open == 100.0
    assert completed.high == 101.5
    assert completed.low == 100.0
    assert completed.close == 101.5
    assert completed.volume == 15
    assert completed.start == base
    assert completed.end == base.replace(second=30)

    # The new bar should start with the last update values.
    next_bar = builder.update(103.0, 3, base.replace(minute=31, second=15))
    assert next_bar is None
    flushed = builder.flush()
    assert flushed is not None
    assert flushed.open == 102.0
    assert flushed.close == 103.0
    assert flushed.volume == 10
