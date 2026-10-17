from __future__ import annotations

import queue
import time
from datetime import datetime, timezone

import pandas as pd
import pytest

from nifty_scalper_bot.data.market_data_hardening import (
    install_market_data_manager_hardening,
)
from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NFO:NIFTY26JUL25000CE"
TOKEN = 987654


def _manager() -> MarketDataManager:
    install_market_data_manager_hardening(MarketDataManager)
    mdm = MarketDataManager(broker=None, websocket=None)
    mdm.register_symbol(SYMBOL, TOKEN)
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


def test_native_mdm_owns_candle_flush_task_lifecycle() -> None:
    ensure_consumer = MarketDataManager._ensure_tick_consumer
    ensure_flush = MarketDataManager._ensure_candle_flush_task
    stop_flush = MarketDataManager._stop_candle_flush_task
    stop = MarketDataManager.stop

    install_market_data_manager_hardening(MarketDataManager)

    assert MarketDataManager._ensure_tick_consumer is ensure_consumer
    assert MarketDataManager._ensure_candle_flush_task is ensure_flush
    assert MarketDataManager._stop_candle_flush_task is stop_flush
    assert MarketDataManager.stop is stop
    assert ensure_consumer.__module__ == "nifty_scalper_bot.data.market_data_manager"
    assert ensure_flush.__module__ == "nifty_scalper_bot.data.market_data_manager"
    assert stop_flush.__module__ == "nifty_scalper_bot.data.market_data_manager"
    assert stop.__module__ == "nifty_scalper_bot.data.market_data_manager"


def test_native_mdm_initializes_candle_flush_lifecycle_state() -> None:
    mdm = _manager()

    assert mdm._candle_flush_task is None
    assert mdm._candle_flush_interval_s >= 0.25
    assert mdm._candle_flush_grace_s >= 0.0


def test_cached_quote_read_does_not_refresh_ingress_age_or_midpoint() -> None:
    mdm = _manager()
    mdm._now_ms = lambda: 2_500.0
    with mdm._lock:
        mdm._tick_cache[SYMBOL] = {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 100.0,
            "bid": 99.5,
            "ask": 100.5,
            "timestamp": "2026-10-07T03:30:00+00:00",
            "source": "ws_full",
        }
        mdm._last_quote_ts_ms[SYMBOL] = 1_000.0
        mdm._last_mid[SYMBOL] = (100.0, 1_000.0)

    quote = mdm.get_quote(SYMBOL)

    assert quote is not None
    assert quote["ltp"] == 100.0
    assert quote["tick_age_ms"] == 1_500.0
    assert quote["quote_age_s"] == 1.5
    assert mdm._last_quote_ts_ms[SYMBOL] == 1_000.0
    assert mdm._last_mid[SYMBOL] == (100.0, 1_000.0)


def test_live_normalization_marks_synthetic_ws_time_non_tradable() -> None:
    mdm = _manager()
    tick = mdm.normalize_live_tick(
        {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 100.0,
            "depth": {
                "buy": [{"price": 99.5, "quantity": 100}],
                "sell": [{"price": 100.5, "quantity": 120}],
            },
        },
        source="ws",
    )

    assert tick is not None
    assert tick["timestamp_quality"] == "synthetic"
    assert tick["source_timestamp_valid"] is False
    assert tick["hard_readiness_eligible"] is False
    assert tick["tradable_quote"] is False


def test_live_normalization_exposes_two_sided_depth_and_microprice() -> None:
    mdm = _manager()
    tick = mdm.normalize_live_tick(
        {
            "symbol": SYMBOL,
            "instrument_token": TOKEN,
            "ltp": 100.0,
            "exchange_timestamp": "2026-10-07T03:30:00Z",
            "depth": {
                "buy": [
                    {"price": 99.5, "quantity": 100},
                    {"price": 99.0, "quantity": 80},
                    {"price": 98.5, "quantity": 70},
                    {"price": 98.0, "quantity": 60},
                    {"price": 97.5, "quantity": 50},
                ],
                "sell": [
                    {"price": 100.5, "quantity": 120},
                    {"price": 101.0, "quantity": 90},
                    {"price": 101.5, "quantity": 80},
                    {"price": 102.0, "quantity": 70},
                    {"price": 102.5, "quantity": 60},
                ],
            },
        },
        source="ws",
    )

    assert tick is not None
    assert tick["depth_two_sided"] is True
    assert tick["depth_complete_5x5"] is True
    assert tick["bid_depth_levels"] == 5
    assert tick["ask_depth_levels"] == 5
    assert tick["bid_depth_qty_5"] == 360
    assert tick["ask_depth_qty_5"] == 420
    assert tick["depth_imbalance_5"] == pytest.approx(-60 / 780)
    assert 99.5 < tick["microprice"] < 100.5
    assert tick["event_timestamp_ms"] > 0
    assert tick["received_timestamp_ms"] > 0
    assert tick["received_monotonic_ns"] > 0
