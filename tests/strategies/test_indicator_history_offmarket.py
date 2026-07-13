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
    captured: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = captured.append  # type: ignore[method-assign]
    logger = logging.getLogger("nifty_scalper_bot.strategies.indicators")
    logger.addHandler(handler)
    with caplog.at_level(
        logging.INFO, logger="nifty_scalper_bot.strategies.indicators"
    ):
        try:
            result = engine.get_history("NSE:NIFTY")
        finally:
            logger.removeHandler(handler)
    assert result == []
    records = list(caplog.records) + captured
    levels = [
        r.levelno for r in records if "indicator_history_missing" in r.getMessage()
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
