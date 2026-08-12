from __future__ import annotations

from collections import deque

import pandas as pd

from nifty_scalper_bot.data.candle_clock_flush_hardening import (
    install_candle_clock_flush_hardening,
)
from nifty_scalper_bot.data.candle_engine import CandleEngine
from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def _tick(symbol: str, *, bucket: str, enqueued: float, timestamp: str) -> dict:
    return {
        "symbol": symbol,
        "last_price": 100.0,
        "timestamp": timestamp,
        "exchange_timestamp": timestamp,
        "_mdm_priority": 2,
        "_mdm_priority_bucket": bucket,
        "_mdm_enqueued_mono": enqueued,
    }


def test_equal_priority_context_batch_prefers_spot_future_over_near_atm() -> None:
    install_candle_clock_flush_hardening(MarketDataManager)
    mdm = MarketDataManager(kite=None)
    mdm._tick_drain_batch_size = 1
    near = "NFO:NIFTY2681124500CE"
    future = "NFO:NIFTY26AUGFUT"
    near_tick = _tick(
        near,
        bucket="near_atm",
        enqueued=1.0,
        timestamp="2026-08-11T12:22:58+05:30",
    )
    future_tick = _tick(
        future,
        bucket="spot_future_context",
        enqueued=2.0,
        timestamp="2026-08-11T12:22:59+05:30",
    )
    mdm._pending_tick_queues[near] = deque([near_tick])
    mdm._pending_tick_queues[future] = deque([future_tick])
    mdm._pending_tick_count = 2

    batch = mdm._pop_pending_tick_batch()

    assert [tick["symbol"] for tick in batch] == [future]
    assert list(mdm._pending_tick_queues[near]) == [near_tick]


def test_clock_flush_waits_for_same_minute_tick_already_pending() -> None:
    install_candle_clock_flush_hardening(MarketDataManager)
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26AUGFUT"
    minute = pd.Timestamp("2026-08-11T12:22:00+05:30")
    engine = CandleEngine(symbol=symbol)
    engine.current_candle = {
        "timestamp": minute,
        "open": 24400.0,
        "high": 24405.0,
        "low": 24395.0,
        "close": 24402.0,
        "volume": 100,
    }
    mdm._engines[symbol] = engine
    pending = _tick(
        symbol,
        bucket="spot_future_context",
        enqueued=1.0,
        timestamp="2026-08-11T12:22:59+05:30",
    )
    mdm._pending_tick_queues[symbol] = deque([pending])
    mdm._pending_tick_count = 1

    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-11T12:23:02+05:30"),
            grace_seconds=1.5,
        )
        == 0
    )
    assert engine.current_candle is not None
    assert engine.latest_finalized_minute() is None

    mdm._pending_tick_queues.clear()
    mdm._pending_tick_count = 0
    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-11T12:23:03+05:30"),
            grace_seconds=1.5,
        )
        == 1
    )
    assert engine.current_candle is None
    assert engine.latest_finalized_minute() == minute


def test_clock_flush_waits_while_same_symbol_tick_is_inflight() -> None:
    install_candle_clock_flush_hardening(MarketDataManager)
    mdm = MarketDataManager(kite=None)
    symbol = "NFO:NIFTY26AUGFUT"
    minute = pd.Timestamp("2026-08-11T12:22:00+05:30")
    engine = CandleEngine(symbol=symbol)
    engine.current_candle = {
        "timestamp": minute,
        "open": 24400.0,
        "high": 24405.0,
        "low": 24395.0,
        "close": 24402.0,
        "volume": 100,
    }
    mdm._engines[symbol] = engine
    mdm._candle_tick_inflight_symbol = symbol

    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-11T12:23:02+05:30"),
            grace_seconds=1.5,
        )
        == 0
    )
    assert engine.current_candle is not None

    mdm._candle_tick_inflight_symbol = None
    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-11T12:23:03+05:30"),
            grace_seconds=1.5,
        )
        == 1
    )


def _engine_for(symbol: str, minute: pd.Timestamp) -> CandleEngine:
    engine = CandleEngine(symbol=symbol)
    engine.current_candle = {
        "timestamp": minute,
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "volume": 100,
    }
    return engine


def test_clock_flush_waits_for_tick_popped_into_drain_batch() -> None:
    """A received tick stays visible to the clock guard after queue pop."""
    install_candle_clock_flush_hardening(MarketDataManager)
    mdm = MarketDataManager(kite=None)
    mdm._tick_drain_batch_size = 1
    symbol = "NFO:NIFTY2681824250PE"
    minute = pd.Timestamp("2026-08-12T15:26:00+05:30")
    engine = _engine_for(symbol, minute)
    mdm._engines[symbol] = engine
    pending = _tick(
        symbol,
        bucket="near_atm",
        enqueued=1.0,
        timestamp="2026-08-12T15:26:59+05:30",
    )
    mdm._pending_tick_queues[symbol] = deque([pending])
    mdm._pending_tick_count = 1

    batch = mdm._pop_pending_tick_batch()
    assert batch == [pending]
    assert symbol not in mdm._pending_tick_queues

    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-12T15:27:02+05:30"),
            grace_seconds=1.5,
        )
        == 0
    )
    assert engine.current_candle is not None
    assert engine.latest_finalized_minute() is None

    mdm._process_queued_tick(batch[0])
    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-12T15:27:03+05:30"),
            grace_seconds=1.5,
        )
        == 1
    )
    assert engine.latest_finalized_minute() == minute


def test_requeued_popped_tick_reservation_is_not_double_counted() -> None:
    """Budget requeue/pop cycles retain one reservation, then release cleanly."""
    install_candle_clock_flush_hardening(MarketDataManager)
    mdm = MarketDataManager(kite=None)
    mdm._tick_drain_batch_size = 1
    symbol = "NFO:NIFTY2681824250CE"
    minute = pd.Timestamp("2026-08-12T15:26:00+05:30")
    engine = _engine_for(symbol, minute)
    mdm._engines[symbol] = engine
    pending = _tick(
        symbol,
        bucket="near_atm",
        enqueued=1.0,
        timestamp="2026-08-12T15:26:59+05:30",
    )
    mdm._pending_tick_queues[symbol] = deque([pending])
    mdm._pending_tick_count = 1

    first_batch = mdm._pop_pending_tick_batch()
    mdm._requeue_unprocessed_ticks(first_batch)
    second_batch = mdm._pop_pending_tick_batch()
    assert second_batch == [pending]
    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-12T15:27:02+05:30"),
            grace_seconds=1.5,
        )
        == 0
    )

    mdm._process_queued_tick(second_batch[0])
    assert (
        mdm.flush_due_candles(
            now=pd.Timestamp("2026-08-12T15:27:03+05:30"),
            grace_seconds=1.5,
        )
        == 1
    )
    assert engine.latest_finalized_minute() == minute
