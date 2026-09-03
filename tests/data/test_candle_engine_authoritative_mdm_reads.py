from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


SYMBOL = "NSE:NIFTY"


def _rows(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 9, 3, 3, 45, tzinfo=timezone.utc)
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


async def test_history_capacity_comes_from_candle_engine_not_tick_cache(monkeypatch) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    mdm = MarketDataManager(broker=None, websocket=None)

    engine = mdm.get_candle_engine(SYMBOL)

    assert mdm._cache_len == 50
    assert mdm.history_capacity_for(SYMBOL) == engine.max_bars
    assert mdm.history_capacity_for(SYMBOL) >= 400

    # Raw-tick retention may change independently at runtime/tests; finalized
    # OHLC capacity must continue to come from the already-owned CandleEngine.
    mdm._cache_len = 1
    assert mdm.history_capacity_for(SYMBOL) == engine.max_bars
    assert mdm.history_capacity_for(SYMBOL) >= 400


async def test_canonical_reads_survive_missing_projection(monkeypatch) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    mdm = MarketDataManager(broker=None, websocket=None)
    rows = _rows(400)
    engine = mdm.get_candle_engine(SYMBOL)
    engine.import_history(pd.DataFrame(rows), mode="bootstrap", source="historical")

    # `_ohlc` is compatibility/diagnostic projection state only. Destroying it
    # must not destroy the canonical CandleEngine history view.
    with mdm._lock:
        mdm._ohlc[SYMBOL] = deque(maxlen=1)

    bars = mdm.get_ohlc_bars(SYMBOL, limit=400)

    assert len(bars) == 400
    assert pd.Timestamp(bars[0]["timestamp"]) == pd.Timestamp(rows[0]["timestamp"])
    assert mdm.get_latest_closed_bar(SYMBOL) is not None
    assert mdm.is_ohlc_ready(SYMBOL, required_bars=400)


async def test_canonical_reads_do_not_mutate_projection(monkeypatch) -> None:
    monkeypatch.setenv("MDM_TICK_CACHE_LEN", "50")
    mdm = MarketDataManager(broker=None, websocket=None)
    rows = _rows(20)
    mdm.get_candle_engine(SYMBOL).import_history(
        pd.DataFrame(rows), mode="bootstrap", source="historical"
    )
    with mdm._lock:
        mdm._ohlc[SYMBOL] = deque([{"sentinel": True}], maxlen=1)

    result = mdm.get_ohlc_bars(SYMBOL, limit=5)

    assert len(result) == 5
    with mdm._lock:
        assert list(mdm._ohlc[SYMBOL]) == [{"sentinel": True}]
