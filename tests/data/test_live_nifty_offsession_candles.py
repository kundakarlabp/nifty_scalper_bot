"""Prevent live NIFTY quote-only observations from creating off-session OHLC."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from nifty_scalper_bot.data.market_data_manager import (
    MarketDataManager,
    _should_skip_live_nifty_candle,
)

IST = ZoneInfo("Asia/Kolkata")
OPTION = "NFO:NIFTY26O1923150CE"


@pytest.mark.parametrize(
    ("local_time", "expected_skip"),
    [
        ((9, 14), True),
        ((9, 15), False),
        ((10, 0), False),
        ((15, 29), False),
        ((15, 30), False),
        ((15, 31), True),
        ((16, 37), True),
        ((16, 58), True),
    ],
)
def test_live_nifty_contract_candle_session_gate(
    monkeypatch: pytest.MonkeyPatch,
    local_time: tuple[int, int],
    expected_skip: bool,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("ALLOW_OFFHOURS_TESTING", raising=False)
    hour, minute = local_time
    stamp = datetime(2026, 10, 16, hour, minute, tzinfo=IST)
    assert _should_skip_live_nifty_candle(OPTION, stamp) is expected_skip


def test_session_gate_is_nifty_specific_and_keeps_nonlive_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    timestamp = datetime(2026, 10, 16, 16, 58, tzinfo=IST)
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    assert _should_skip_live_nifty_candle(OPTION, timestamp) is False
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    assert _should_skip_live_nifty_candle("NFO:RELIANCE26OCTFUT", timestamp) is False
    assert _should_skip_live_nifty_candle("NFO:NIFTY26OCTFUT", timestamp) is True
    assert _should_skip_live_nifty_candle("NSE:NIFTY", timestamp) is True


def test_offsession_live_tick_preserves_quote_but_not_ohlc(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm.register_symbol("NSE:NIFTY", 256265)
    engine = mdm.get_candle_engine("NSE:NIFTY")
    tick = {
        "instrument_token": 256265,
        "symbol": "NSE:NIFTY",
        "last_price": 100.0,
        "exchange_timestamp": datetime(2026, 10, 16, 16, 58, tzinfo=IST),
    }
    mdm._process_queued_tick(tick)
    latest = mdm.get_latest_tick("NSE:NIFTY")
    assert latest is not None
    assert latest["last_price"] == 100.0
    assert engine.current_candle is None
    assert engine.get_completed_bars() == []


def test_regular_session_live_tick_still_builds_candle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm.register_symbol("NSE:NIFTY", 256265)
    engine = mdm.get_candle_engine("NSE:NIFTY")
    mdm._process_queued_tick(
        {
            "instrument_token": 256265,
            "symbol": "NSE:NIFTY",
            "last_price": 100.0,
            "exchange_timestamp": datetime(2026, 10, 16, 10, 15, tzinfo=IST),
        }
    )
    assert engine.current_candle is not None
    assert float(engine.current_candle["close"]) == 100.0
