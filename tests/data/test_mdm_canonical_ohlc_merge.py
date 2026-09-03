from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


SYMBOL = "NFO:NIFTY26JUN24000CE"


def _rows(count: int) -> list[dict[str, object]]:
    start = datetime(2026, 1, 1, 3, 45, tzinfo=timezone.utc)
    return [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 100.0 + index,
            "high": 101.0 + index,
            "low": 99.0 + index,
            "close": 100.5 + index,
            "volume": 1_000 + index,
        }
        for index in range(count)
    ]


def _mdm_with_history(count: int = 50) -> tuple[MarketDataManager, list[dict[str, object]]]:
    mdm = MarketDataManager(broker=None, websocket=None)
    rows = _rows(count)
    engine = mdm.get_candle_engine(SYMBOL)
    engine.import_history(pd.DataFrame(rows), mode="bootstrap", source="historical")
    return mdm, rows


def test_canonical_getter_reads_candle_engine_and_ignores_projection_overlap() -> None:
    mdm, rows = _mdm_with_history(50)

    # `_ohlc` is compatibility/diagnostic projection state only. A stale legacy
    # projection with overlapping timestamps and different prices must not alter
    # the canonical CandleEngine view.
    with mdm._lock:
        mdm._ohlc[SYMBOL] = deque(
            [
                {
                    **rows[index],
                    "close": 1_000.0 + index,
                    "source": "legacy_projection",
                }
                for index in range(35, 50)
            ],
            maxlen=250,
        )

    bars = mdm.get_ohlc_bars(SYMBOL)

    assert len(bars) == 50
    assert [pd.Timestamp(bar["timestamp"]) for bar in bars] == [
        pd.Timestamp(row["timestamp"]) for row in rows
    ]
    assert bars[-1]["close"] == rows[-1]["close"]


def test_projection_provisional_row_cannot_overwrite_finalized_canonical_bar() -> None:
    mdm, rows = _mdm_with_history(2)
    with mdm._lock:
        mdm._ohlc[SYMBOL] = deque(
            [
                {
                    **rows[-1],
                    "close": 999.0,
                    "provisional": True,
                    "source": "legacy_projection",
                }
            ],
            maxlen=250,
        )

    bars = mdm.get_ohlc_bars(SYMBOL)

    assert len(bars) == 2
    assert bars[-1]["close"] == rows[-1]["close"]


def test_canonical_getter_applies_limit_after_candle_engine_read() -> None:
    mdm, rows = _mdm_with_history(10)
    projection_before = list(mdm._ohlc.get(SYMBOL, ()))

    bars = mdm.get_ohlc_bars(SYMBOL, limit=3)

    assert len(bars) == 3
    assert [pd.Timestamp(bar["timestamp"]) for bar in bars] == [
        pd.Timestamp(row["timestamp"]) for row in rows[-3:]
    ]
    assert list(mdm._ohlc.get(SYMBOL, ())) == projection_before


def test_canonical_getter_is_deterministic_and_returns_defensive_rows() -> None:
    mdm, rows = _mdm_with_history(5)

    first = mdm.get_ohlc_bars(SYMBOL)
    second = mdm.get_ohlc_bars(SYMBOL)

    assert first == second
    first[-1]["close"] = -1.0
    third = mdm.get_ohlc_bars(SYMBOL)
    assert third[-1]["close"] == rows[-1]["close"]


def test_projection_only_state_is_not_canonical_for_fully_initialized_mdm() -> None:
    mdm = MarketDataManager(broker=None, websocket=None)
    with mdm._lock:
        mdm._ohlc[SYMBOL] = deque(
            [
                {
                    "timestamp": datetime(2026, 1, 1, 3, 45, tzinfo=timezone.utc),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "volume": 1_000,
                }
            ],
            maxlen=250,
        )

    assert mdm.get_ohlc_bars(SYMBOL) == []
    assert mdm.get_latest_closed_bar(SYMBOL) is None


def test_canonical_getter_limit_zero_and_negative_limit() -> None:
    mdm, _rows_data = _mdm_with_history(1)

    assert mdm.get_ohlc_bars(SYMBOL, limit=0) == []
    with pytest.raises(ValueError):
        mdm.get_ohlc_bars(SYMBOL, limit=-1)
