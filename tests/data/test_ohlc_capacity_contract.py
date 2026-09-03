from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.ohlc_capacity_contract import (
    configured_ohlc_capacity,
    install_mdm_ohlc_capacity_contract,
)


SYMBOL = "NSE:NIFTY"


def _bars(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 3, 3, 45, tzinfo=timezone.utc)  # 09:15 IST
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 24_000.0 + index,
            "high": 24_001.0 + index,
            "low": 23_999.0 + index,
            "close": 24_000.5 + index,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def test_completed_ohlc_capacity_has_full_session_floor(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "250")

    assert configured_ohlc_capacity() == 400


def test_raw_tick_cache_stays_small_while_completed_ohlc_is_deeper(monkeypatch) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()

    mdm = MarketDataManager(broker=None, websocket=None)

    assert mdm._cache_len == 50
    assert mdm._raw_tick_history[SYMBOL].maxlen == 50
    assert mdm._ohlc[SYMBOL].maxlen == 500
    engine = mdm._get_engine(SYMBOL)
    assert engine.max_bars >= 500
    assert engine._completed_candles.maxlen >= 500


def test_canonical_read_keeps_opening_range_after_250_bar_boundary(monkeypatch) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    engine = mdm._get_engine(SYMBOL)
    rows = _bars(400)

    # Reproduce the old split state: CandleEngine has the full session while
    # the compatibility MDM projection contains only the latest 250 rows.
    engine._completed_candles = deque(rows, maxlen=500)
    mdm._ohlc[SYMBOL] = deque(rows[-250:], maxlen=250)

    full_session = mdm.get_ohlc_bars(SYMBOL, limit=400)

    assert len(full_session) == 400
    assert full_session[0]["timestamp"] == rows[0]["timestamp"]
    assert mdm._ohlc[SYMBOL].maxlen == 500
    assert mdm._raw_tick_history[SYMBOL].maxlen == 50


def test_small_history_requests_remain_bounded(monkeypatch) -> None:
    monkeypatch.setenv("MDM_OHLC_CACHE_LEN", "500")
    install_mdm_ohlc_capacity_contract()
    mdm = MarketDataManager(broker=None, websocket=None)
    engine = mdm._get_engine(SYMBOL)
    rows = _bars(400)
    engine._completed_candles = deque(rows, maxlen=500)

    result = mdm.get_ohlc_bars(SYMBOL, limit=30)

    assert len(result) == 30
    assert result[0]["timestamp"] == rows[-30]["timestamp"]
