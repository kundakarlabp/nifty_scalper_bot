from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_has_min_bars_rejects_non_monotonic_timestamps() -> None:
    engine = IndicatorEngine()
    symbol = "NIFTY25JAN25000CE"
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(symbol, 101.0, timestamp=ts)

    assert engine.has_min_bars(symbol, 2) is False


def test_has_min_bars_rejects_missing_candle_gap_for_live_feed_jitter() -> None:
    engine = IndicatorEngine()
    symbol = "NIFTY25JAN25000PE"
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(symbol, 101.0, timestamp=ts + timedelta(minutes=3))

    assert engine.has_min_bars(symbol, 2) is False


def test_has_min_bars_rejects_large_gap_after_integrity_warning() -> None:
    engine = IndicatorEngine()
    symbol = "NIFTY25JAN25050PE"
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(symbol, 101.0, timestamp=ts + timedelta(minutes=15))

    assert engine.has_min_bars(symbol, 2) is False


def test_has_min_bars_rejects_provisional_candles() -> None:
    engine = IndicatorEngine()
    symbol = "NIFTY25JAN25100CE"
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(
        symbol,
        101.0,
        timestamp=ts + timedelta(minutes=1),
        is_provisional=True,
    )

    assert engine.has_min_bars(symbol, 2) is False


def test_has_min_bars_rejects_incomplete_candles() -> None:
    engine = IndicatorEngine()
    symbol = "NIFTY25JAN25100PE"
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(
        symbol,
        101.0,
        timestamp=ts + timedelta(minutes=1),
        is_complete=False,
    )

    assert engine.has_min_bars(symbol, 2) is False
