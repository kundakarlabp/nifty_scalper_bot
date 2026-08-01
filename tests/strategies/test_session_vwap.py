"""VWAP must be session-anchored and honestly volume-weighted (P1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine

_IST = timezone(timedelta(hours=5, minutes=30))


def _engine_with_bars(bars: list[tuple[datetime, float, int]]) -> IndicatorEngine:
    engine = IndicatorEngine()
    for stamp, price, volume in bars:
        engine.update_price(
            "NFO:NIFTY2680424400CE",
            {"open": price, "high": price, "low": price, "close": price},
            volume=volume,
            timestamp=stamp,
        )
    return engine


def _session_bar(minute: int, price: float, volume: int):
    stamp = datetime(2026, 8, 3, 9, 15, tzinfo=_IST) + timedelta(minutes=minute)
    return stamp, price, volume


def test_session_vwap_is_volume_weighted() -> None:
    engine = _engine_with_bars(
        [_session_bar(0, 100.0, 100), _session_bar(1, 200.0, 300)]
    )

    # (100*100 + 200*300) / 400
    assert engine.get_session_vwap("NFO:NIFTY2680424400CE") == pytest.approx(175.0)


def test_session_vwap_differs_from_rolling_vwap() -> None:
    bars = [_session_bar(i, 100.0 + i, 10) for i in range(30)]
    engine = _engine_with_bars(bars)
    symbol = "NFO:NIFTY2680424400CE"

    rolling = engine.get_vwap(symbol)
    session = engine.get_session_vwap(symbol)

    assert rolling is not None and session is not None
    # The rolling 20-bar mean ignores the first 10 bars of the session.
    assert session < rolling


def test_zero_volume_series_yields_no_vwap() -> None:
    engine = _engine_with_bars([_session_bar(i, 100.0 + i, 0) for i in range(5)])

    assert engine.get_session_vwap("NFO:NIFTY2680424400CE") is None


def test_unvolumed_bars_do_not_get_fake_weights() -> None:
    engine = IndicatorEngine()
    # Only the second bar carries volume, so it alone defines VWAP.
    value = engine._calculate_vwap([100.0, 200.0], [0, 50])

    assert value == pytest.approx(200.0)


def test_vwap_pro_prefers_the_exchange_reference() -> None:
    import inspect

    from nifty_scalper_bot.strategies.elite_strategies import vwap_pro

    source = inspect.getsource(vwap_pro.VWAPProStrategy._evaluate_signal)
    exchange = source.index("exchange_vwap")
    session = source.index("session_vwap")
    rolling = source.index("indicators.get('vwap')")
    assert exchange < session < rolling
