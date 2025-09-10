from datetime import datetime, timezone

from src.data.source import MinuteBarBuilder


def test_minute_bar_builder_across_minutes() -> None:
    builder = MinuteBarBuilder(max_bars=5)

    t1 = datetime(2025, 1, 1, 9, 30, 15, tzinfo=timezone.utc)
    builder.on_tick({'last_price': 100, 'volume': 10, 'exchange_timestamp': t1})
    builder.on_tick({'last_price': 102, 'volume': 5, 'exchange_timestamp': t1.replace(second=30)})
    builder.on_tick({'last_price': 99, 'volume': 8, 'exchange_timestamp': t1.replace(second=45)})

    t2 = datetime(2025, 1, 1, 9, 31, 5, tzinfo=timezone.utc)
    builder.on_tick({'last_price': 105, 'volume': 12, 'exchange_timestamp': t2})
    builder.on_tick({'last_price': 107, 'volume': 3, 'exchange_timestamp': t2.replace(second=20)})

    bars = builder.get_recent_bars(5)
    assert len(bars) == 2
    assert bars[0]['open'] == 100
    assert bars[0]['high'] == 102
    assert bars[0]['low'] == 99
    assert bars[0]['close'] == 99
    assert bars[0]['volume'] == 23
    # (100*10 + 102*5 + 99*8) / 23 = 100.0869
    assert abs(bars[0]['vwap'] - 100.0869) < 0.01

    assert bars[1]['open'] == 105
    assert bars[1]['high'] == 107
    assert bars[1]['low'] == 105
    assert bars[1]['close'] == 107
    assert bars[1]['volume'] == 15


def test_have_min_bars() -> None:
    builder = MinuteBarBuilder()
    t = datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    builder.on_tick({'last_price': 100, 'exchange_timestamp': t})
    assert not builder.have_min_bars(2)
    t2 = t.replace(minute=31)
    builder.on_tick({'last_price': 101, 'exchange_timestamp': t2})
    assert builder.have_min_bars(2)


def test_tick_without_last_price_is_ignored() -> None:
    builder = MinuteBarBuilder()
    t = datetime(2025, 1, 1, 9, 30, 0, tzinfo=timezone.utc)
    builder.on_tick({'volume': 5, 'exchange_timestamp': t})
    assert builder.get_recent_bars(1) == []

    builder.on_tick({'last_price': 100, 'volume': 10, 'exchange_timestamp': t})
    bars = builder.get_recent_bars(1)
    assert len(bars) == 1
    assert bars[0]['open'] == 100
