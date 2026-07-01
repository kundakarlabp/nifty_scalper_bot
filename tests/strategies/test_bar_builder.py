from datetime import datetime, timedelta, timezone

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


def test_late_tick_cannot_rewind_active_bar() -> None:
    builder = OneMinuteBarBuilder()
    base = datetime(2026, 1, 2, 9, 15, 30, tzinfo=timezone.utc)
    assert builder.update(100.0, 1, base) is None
    assert builder.update(90.0, 1, base - timedelta(minutes=1)) is None
    closed = builder.update(101.0, 1, base + timedelta(minutes=1))
    assert closed is not None
    assert closed.start == base.replace(second=0, microsecond=0)
    assert closed.open == 100.0
    assert closed.low == 100.0


def test_bar_timestamp_is_utc_minute_bucket_not_first_tick_second() -> None:
    builder = OneMinuteBarBuilder()
    tick_ts = datetime(2026, 1, 2, 9, 15, 42, tzinfo=timezone.utc)
    builder.update(100.0, 1, tick_ts)
    closed = builder.update(101.0, 1, tick_ts + timedelta(seconds=18))
    assert closed is not None
    assert closed.timestamp == datetime(2026, 1, 2, 9, 15, tzinfo=timezone.utc)
