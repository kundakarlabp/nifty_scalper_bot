from __future__ import annotations

from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.strategies.indicators import IndicatorEngine


def test_has_min_bars_rejects_non_monotonic_timestamps() -> None:
    engine = IndicatorEngine()
    symbol = 'NIFTY25JAN25000CE'
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(symbol, 101.0, timestamp=ts)

    assert engine.has_min_bars(symbol, 2) is False


def test_has_min_bars_rejects_missing_candle_gap() -> None:
    engine = IndicatorEngine()
    symbol = 'NIFTY25JAN25000PE'
    ts = datetime.now(timezone.utc).replace(second=0, microsecond=0)
    engine.update_price(symbol, 100.0, timestamp=ts)
    engine.update_price(symbol, 101.0, timestamp=ts + timedelta(minutes=3))

    assert engine.has_min_bars(symbol, 2) is False
