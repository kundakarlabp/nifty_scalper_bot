from collections import deque
from datetime import datetime, timedelta
import logging
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from nifty_scalper_bot.data.pipeline import Candle, CandleStore, MarketDataPipeline
from nifty_scalper_bot.data.source import DataIntegrityError
from nifty_scalper_bot.data.validator import validate_tick

IST = ZoneInfo("Asia/Kolkata")


def test_naive_tick_timestamp_is_local_ist() -> None:
    tick = validate_tick({"symbol": "NFO:TEST", "timestamp": "2026-01-02 11:55:00", "ltp": 100})
    assert tick.timestamp.isoformat() == "2026-01-02T11:55:00+05:30"


def test_naive_candle_timestamp_is_local_ist() -> None:
    store = CandleStore()
    store.push(Candle("NFO:TEST", datetime(2026, 1, 2, 11, 55), 100, 100, 100, 100, 1))
    assert store.get("NFO:TEST")[0].timestamp.isoformat() == "2026-01-02T11:55:00+05:30"


def test_future_candle_is_rejected_without_store_write() -> None:
    store = CandleStore()
    ts = datetime.now(IST) + timedelta(minutes=10)
    with pytest.raises(DataIntegrityError):
        store.push(Candle("NFO:TEST", ts, 100, 100, 100, 100, 1))
    assert store.get("NFO:TEST") == []


def test_one_minute_bar_builder_localizes_naive_ist_to_utc() -> None:
    from nifty_scalper_bot.strategies.bar_builder import OneMinuteBarBuilder

    builder = OneMinuteBarBuilder()
    builder.update(100, 1, datetime(2026, 1, 2, 12, 52))
    closed = builder.update(101, 1, datetime(2026, 1, 2, 12, 53))

    assert closed is not None
    assert closed.timestamp.isoformat() == "2026-01-02T07:22:00+00:00"


def test_candle_builder_quarantines_future_last_ts_then_accepts_current_bar() -> None:
    from nifty_scalper_bot.data.pipeline import CandleBuilder, ValidatedTick

    builder = CandleBuilder()
    sym = "NFO:TEST"
    builder._last_ts[sym] = pd.Timestamp.now(tz=IST) + pd.Timedelta(minutes=10)
    now = pd.Timestamp.now(tz=IST).floor("1min")

    assert builder.on_tick(ValidatedTick(sym, now, 100, 1)) is None
    closed = builder.on_tick(ValidatedTick(sym, now + pd.Timedelta(minutes=1), 101, 1))

    assert closed is not None
    assert closed.timestamp == now


def test_candle_store_quarantines_future_tail_before_valid_current_bar() -> None:
    store = CandleStore()
    sym = "NFO:TEST"
    future = pd.Timestamp.now(tz=IST) + pd.Timedelta(minutes=10)
    current = pd.Timestamp.now(tz=IST).floor("1min")
    store._store[sym] = deque([Candle(sym, future, 100, 100, 100, 100, 1)])

    store.push(Candle(sym, current, 101, 101, 101, 101, 1))

    candles = store.get(sym)
    assert len(candles) == 1
    assert candles[0].timestamp.floor("1min") == current


def test_candle_store_out_of_order_log_has_forensic_fields(caplog: pytest.LogCaptureFixture) -> None:
    store = CandleStore()
    sym = "NFO:TEST"
    base = pd.Timestamp.now(tz=IST).floor("1min") - pd.Timedelta(minutes=10)
    last = base + pd.Timedelta(minutes=5)
    incoming = base
    store.push(Candle(sym, last, 105, 106, 104, 105, 7))

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.data.pipeline"):
        with pytest.raises(DataIntegrityError):
            store.push(Candle(sym, incoming, 101, 102, 100, 101, 3))

    record = next(rec for rec in caplog.records if getattr(rec, "event", "") == "candle_store_out_of_order")
    assert record.symbol == sym
    assert record.incoming_ts == incoming.isoformat()
    assert record.last_ts == last.isoformat()
    assert record.incoming_close == 101.0
    assert record.last_close == 105.0
    assert record.store_size == 1
    assert record.source == "candle_store"
    assert record.reason == "incoming_before_last_store_ts"


def test_pipeline_flush_future_candle_is_rejected_without_raise() -> None:
    pipeline = MarketDataPipeline()
    sym = "NFO:TEST"
    future = pd.Timestamp.now(tz=IST) + pd.Timedelta(minutes=10)
    pipeline.builder._active[sym] = {
        "symbol": sym,
        "timestamp": future,
        "open": 100.0,
        "high": 100.0,
        "low": 100.0,
        "close": 100.0,
        "volume": 1.0,
    }

    assert pipeline.flush(sym) is None
    assert pipeline.get_candles(sym) == []
