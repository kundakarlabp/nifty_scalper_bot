"""Test that indicator history-missing logs are throttled and downgraded
when the market is closed."""

from __future__ import annotations

import logging

import pytest

from nifty_scalper_bot.strategies.indicators import IndicatorEngine
from nifty_scalper_bot.utils import logging as nsb_logging


@pytest.fixture(autouse=True)
def _reset_throttle_state():
    """Args: none. Returns: None. Raises: none."""
    with nsb_logging._THROTTLE_LOCK:
        nsb_logging._THROTTLE_STATE.clear()
    yield
    with nsb_logging._THROTTLE_LOCK:
        nsb_logging._THROTTLE_STATE.clear()


def test_indicator_history_missing_market_open_emits_info(monkeypatch, caplog):
    """Args: monkeypatch, caplog. Returns: None. Raises: AssertionError."""
    from nifty_scalper_bot.utils import market_hours

    monkeypatch.setattr(market_hours, "is_market_open_now", lambda: True)
    engine = IndicatorEngine()
    with caplog.at_level(logging.INFO):
        result = engine.get_history("NSE:NIFTY")

    assert result == []
    levels = [
        record.levelno
        for record in caplog.records
        if "indicator_history_missing" in record.getMessage()
    ]
    assert any(level >= logging.INFO for level in levels), levels


def test_indicator_history_missing_market_closed_is_debug(monkeypatch, caplog):
    """Args: monkeypatch, caplog. Returns: None. Raises: AssertionError."""
    from nifty_scalper_bot.utils import market_hours

    monkeypatch.setattr(market_hours, "is_market_open_now", lambda: False)
    engine = IndicatorEngine()
    with caplog.at_level(
        logging.DEBUG, logger="nifty_scalper_bot.strategies.indicators"
    ):
        engine.get_history("NSE:NIFTY")

    info_records = [
        r
        for r in caplog.records
        if r.levelno >= logging.INFO and "indicator_history_missing" in r.getMessage()
    ]
    assert info_records == [], info_records


def test_missing_indicator_history_emits_once_until_recovered(
    monkeypatch, caplog
) -> None:
    from datetime import datetime, timezone

    from nifty_scalper_bot.utils import market_hours

    monkeypatch.setattr(market_hours, "is_market_open_now", lambda: True)
    engine = IndicatorEngine()
    symbol = "NFO:NIFTY26O1923150CE"
    with caplog.at_level(logging.INFO):
        assert engine.get_history(symbol) == []
        assert engine.get_history(symbol) == []
        assert engine.get_history(symbol, field="bars") == []
        missing = [
            item for item in caplog.records
            if getattr(item, "event", "") == "indicator_engine_history_missing"
        ]
        assert len(missing) == 1
        assert symbol in missing[0].getMessage()

        engine.update_price(
            symbol,
            103.5,
            timestamp=datetime(2026, 10, 16, 5, 1, tzinfo=timezone.utc),
        )
        assert engine.history_count(symbol) == 1
        assert engine.get_history(symbol) == [103.5]
        recovered = [
            item for item in caplog.records
            if getattr(item, "event", "") == "INDICATOR_HISTORY_RECOVERED"
        ]
        assert len(recovered) == 1

        # A genuinely new missing-history episode must remain observable.
        with engine._lock:
            engine._histories.pop(symbol)
        assert engine.get_history(symbol) == []
        assert len([
            item for item in caplog.records
            if getattr(item, "event", "") == "indicator_engine_history_missing"
        ]) == 2


def test_reseed_resolves_missing_history_episode(monkeypatch, caplog) -> None:
    from nifty_scalper_bot.utils import market_hours

    monkeypatch.setattr(market_hours, "is_market_open_now", lambda: True)
    engine = IndicatorEngine()
    symbol = "NFO:NIFTY26O1923150PE"
    assert engine.get_history(symbol) == []
    with caplog.at_level(logging.INFO):
        result = engine.replace_history(
            symbol,
            [{
                "timestamp": "2026-10-16T05:02:00+00:00",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "volume": 10,
            }],
        )
    assert result == 1
    assert engine.history_count(symbol) == 1
    assert len([
        item for item in caplog.records
        if getattr(item, "event", "") == "INDICATOR_HISTORY_RECOVERED"
    ]) == 1
