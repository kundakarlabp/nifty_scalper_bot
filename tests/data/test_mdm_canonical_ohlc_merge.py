from __future__ import annotations

from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
import threading

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _mdm() -> MarketDataManager:
    mdm = MarketDataManager.__new__(MarketDataManager)
    mdm._lock = threading.RLock()
    mdm._cache_len = 250
    mdm._ohlc = defaultdict(lambda: deque(maxlen=mdm._cache_len))
    return mdm


def _bar(
    i: int,
    *,
    source: str = "historical",
    provisional: bool = False,
    close: float | None = None,
) -> dict:
    ts = datetime(2026, 1, 1, 9, 15, tzinfo=timezone.utc) + timedelta(minutes=i)
    return {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "timestamp": ts,
        "open": 100 + i,
        "high": 101 + i,
        "low": 99 + i,
        "close": float(close if close is not None else 100 + i),
        "volume": i,
        "source": source,
        "provisional": provisional,
        "synthetic": False,
        "timestamp_quality": "broker",
    }


def test_canonical_getter_merges_history_and_live_dedupes_overlaps() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    for i in range(50):
        mdm._ohlc[key].append(_bar(i, source="historical"))
    for i in range(35, 50):
        mdm._ohlc[symbol].append(_bar(i, source="ws_candle", close=1000 + i))

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 50
    assert [bar["timestamp"] for bar in bars] == sorted(
        bar["timestamp"] for bar in bars
    )
    assert len({bar["timestamp"] for bar in bars}) == 50
    assert bars[-1]["close"] == 1049.0


def test_canonical_getter_does_not_let_provisional_current_candle_overwrite_history() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(_bar(1, source="historical", close=101.0))
    mdm._ohlc[key].append(
        _bar(1, source="ws_candle", provisional=True, close=999.0)
    )

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 1
    assert bars[0]["close"] == 101.0
    assert bars[0]["source"] == "historical"


def test_completed_live_candle_precedes_historical_on_same_timestamp() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    mdm._ohlc[key].append(_bar(2, source="historical", close=102.0))
    mdm._ohlc[key].append(_bar(2, source="ws_candle", close=202.0))

    bars = mdm.get_ohlc_bars(symbol)

    assert len(bars) == 1
    assert bars[0]["close"] == 202.0
    assert bars[0]["source"] == "ws_candle"


def test_canonical_getter_applies_limit_after_merge_and_does_not_mutate_storage() -> None:
    mdm = _mdm()
    symbol = "NFO:NIFTY26JUN24000CE"
    key = mdm._bar_symbol_key(symbol)
    for i in range(10):
        mdm._ohlc[key].append(_bar(i, source="historical"))
    before = list(mdm._ohlc[key])

    bars = mdm.get_ohlc_bars(symbol, limit=3)

    assert [bar["timestamp"] for bar in bars] == [
        before[-3]["timestamp"],
        before[-2]["timestamp"],
        before[-1]["timestamp"],
    ]
    assert list(mdm._ohlc[key]) == before
