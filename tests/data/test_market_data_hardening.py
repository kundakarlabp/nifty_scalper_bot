from __future__ import annotations

import queue
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from nifty_scalper_bot.data.market_data_hardening import (
    install_market_data_manager_hardening,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager
from nifty_scalper_bot.data.required_tick_backlog_hardening import (
    install_required_tick_backlog_hardening,
)

SYMBOL = "NFO:NIFTY26JUL25000CE"
TOKEN = 987654
FUTURE = "NFO:NIFTY26JULFUT"
FUTURE_TOKEN = 987655


class _NoopLoop:
    def is_running(self) -> bool:
        return True

    def call_soon_threadsafe(self, callback, *args) -> None:
        del callback, args


def _manager() -> MarketDataManager:
    install_market_data_manager_hardening(MarketDataManager)
    install_required_tick_backlog_hardening(MarketDataManager)
    mdm = MarketDataManager(broker=None, websocket=None)
    mdm.register_symbol(SYMBOL, TOKEN)
    return mdm


def _future_manager() -> MarketDataManager:
    mdm = _manager()
    mdm.register_symbol(FUTURE, FUTURE_TOKEN)
    mdm.set_active_contract_basket(
        {
            "spot_symbol": "NSE:NIFTY",
            "futures_symbol": FUTURE,
            "selected_ce": SYMBOL,
            "selected_pe": None,
            "option_symbols": [SYMBOL],
            "symbols": ["NSE:NIFTY", FUTURE, SYMBOL],
            "token_by_symbol": {SYMBOL: TOKEN, FUTURE: FUTURE_TOKEN},
            "all_tokens": [TOKEN, FUTURE_TOKEN],
        }
    )
    return mdm


def test_ws_tick_without_broker_timestamp_is_rejected_for_candles() -> None:
    mdm = _manager()

    normalized = mdm._normalize_ws_tick(
        {"instrument_token": TOKEN, "last_price": 100.0}
    )

    assert normalized is None
    assert mdm._candle_metrics["invalid_candle_timestamp_total"] == 1


def test_synthetic_timestamp_does_not_pass_fresh_ws_ltp() -> None:
    mdm = _manager()
    with mdm._lock:
        mdm._latest_ticks[SYMBOL] = {
            "symbol": SYMBOL,
            "source": "ws",
            "ltp": 100.0,
            "timestamp": time.time(),
            "timestamp_quality": "synthetic",
        }
        mdm._last_tick_source[SYMBOL] = "ws"

    assert mdm.has_fresh_ws_ltp([SYMBOL], max_age_seconds=5.0) is False


def test_exchange_timestamp_passes_fresh_ws_ltp() -> None:
    mdm = _manager()
    ts = datetime.now(timezone.utc)
    with mdm._lock:
        mdm._latest_ticks[SYMBOL] = {
            "symbol": SYMBOL,
            "source": "ws",
            "ltp": 100.0,
            "exchange_timestamp": ts,
            "timestamp": ts,
            "timestamp_quality": "exchange",
        }
        mdm._last_tick_source[SYMBOL] = "ws"

    assert mdm.has_fresh_ws_ltp([SYMBOL], max_age_seconds=5.0) is True


def test_fallback_ingress_uses_thread_safe_queue() -> None:
    mdm = _manager()

    assert isinstance(mdm._fallback_tick_queue, queue.Queue)


def test_fallback_queue_coalesces_same_symbol_when_full() -> None:
    mdm = _manager()
    mdm._fallback_tick_queue = queue.Queue(maxsize=1)

    assert (
        mdm._put_fallback_tick_nowait(
            {"symbol": SYMBOL, "instrument_token": TOKEN, "last_price": 101.0}
        )
        is True
    )
    assert (
        mdm._put_fallback_tick_nowait(
            {"symbol": SYMBOL, "instrument_token": TOKEN, "last_price": 102.0}
        )
        is True
    )

    retained = mdm._fallback_tick_queue.get_nowait()
    assert retained["last_price"] == 102.0


def test_required_future_burst_is_compacted_and_bounded() -> None:
    mdm = _future_manager()
    mdm._tick_queue_maxsize = 16
    loop = _NoopLoop()
    ts = datetime.now(timezone.utc).replace(second=10, microsecond=0)

    for index in range(10_000):
        mdm._enqueue_latest_tick_for_drain(
            {
                "symbol": FUTURE,
                "instrument_token": FUTURE_TOKEN,
                "last_price": 24000.0 + float(index % 9),
                "exchange_timestamp": ts,
                "timestamp": ts,
                "volume_traded_today": 1000 + index,
            },
            loop,
        )

    stats = mdm.get_tick_pressure_stats()
    assert stats["pending_ticks"] <= 16
    assert stats["coalesced_total"] > 0
    assert stats["unexplained_loss"] == 0


def test_compacted_future_replay_preserves_ohlc_and_latest_volume() -> None:
    mdm = _future_manager()
    mdm._tick_queue_maxsize = 8
    loop = _NoopLoop()
    ts = datetime.now(timezone.utc).replace(second=10, microsecond=0)

    for price, volume in zip(
        [24100.0, 24130.0, 24080.0, 24120.0, 24110.0, 24120.0],
        [1000, 1050, 1075, 1100, 1110, 1120],
        strict=True,
    ):
        mdm._enqueue_latest_tick_for_drain(
            {
                "symbol": FUTURE,
                "instrument_token": FUTURE_TOKEN,
                "last_price": price,
                "exchange_timestamp": ts,
                "timestamp": ts,
                "volume_traded_today": volume,
            },
            loop,
        )

    for raw in mdm._pop_pending_tick_batch():
        mdm._process_queued_tick(raw)
        mdm._tick_processed_total += 1

    candle = mdm._engines[FUTURE].current_candle
    assert candle["open"] == 24100.0
    assert candle["high"] == 24130.0
    assert candle["low"] == 24080.0
    assert candle["close"] == 24120.0
    latest = mdm.get_latest_tick(FUTURE)
    assert latest["volume_cumulative"] == 1120
    assert mdm.get_tick_pressure_stats()["unexplained_loss"] == 0


def test_compaction_keeps_adjacent_minutes_distinct() -> None:
    mdm = _future_manager()
    mdm._tick_queue_maxsize = 8
    loop = _NoopLoop()
    first = datetime.now(timezone.utc).replace(second=10, microsecond=0)
    second = first + timedelta(minutes=1)

    for ts, price in (
        (first, 24100.0),
        (first, 24120.0),
        (first, 24090.0),
        (second, 24110.0),
        (second, 24130.0),
        (second, 24100.0),
    ):
        mdm._enqueue_latest_tick_for_drain(
            {
                "symbol": FUTURE,
                "instrument_token": FUTURE_TOKEN,
                "last_price": price,
                "exchange_timestamp": ts,
                "timestamp": ts,
            },
            loop,
        )

    queue_for_future = mdm._pending_tick_queues[FUTURE]
    minute_keys = {
        pd.Timestamp(item["exchange_timestamp"]).floor("1min")
        for item in queue_for_future
    }
    assert len(minute_keys) == 2


def test_selected_option_compaction_preserves_latest_depth() -> None:
    mdm = _future_manager()
    mdm._tick_queue_maxsize = 16
    loop = _NoopLoop()
    ts = datetime.now(timezone.utc).replace(second=10, microsecond=0)

    for index in range(12):
        bid = 100.0 + index
        ask = bid + 0.5
        mdm._enqueue_latest_tick_for_drain(
            {
                "symbol": SYMBOL,
                "instrument_token": TOKEN,
                "last_price": bid + 0.25,
                "exchange_timestamp": ts,
                "timestamp": ts,
                "depth": {
                    "buy": [{"price": bid, "quantity": 100}],
                    "sell": [{"price": ask, "quantity": 100}],
                },
            },
            loop,
        )

    for raw in mdm._pop_pending_tick_batch():
        mdm._process_queued_tick(raw)
        mdm._tick_processed_total += 1

    latest = mdm.get_latest_tick(SYMBOL)
    assert latest["bid"] == 111.0
    assert latest["ask"] == 111.5


def test_clock_flush_finalizes_idle_candle_without_next_tick() -> None:
    mdm = _manager()
    engine = mdm._get_engine(SYMBOL)
    candle_minute = pd.Timestamp.now(tz="Asia/Kolkata").floor("1min") - pd.Timedelta(
        minutes=2
    )
    engine.current_candle = {
        "timestamp": candle_minute,
        "open": 100.0,
        "high": 103.0,
        "low": 99.0,
        "close": 102.0,
        "volume": 10.0,
    }

    flushed = mdm.flush_due_candles(
        now=candle_minute + pd.Timedelta(minutes=2),
        grace_seconds=0.0,
    )

    assert flushed == 1
    assert engine.current_candle is None
    assert len(mdm._ohlc[SYMBOL]) == 1
    assert mdm._ohlc[SYMBOL][0]["source"] == "clock_flush_candle"
