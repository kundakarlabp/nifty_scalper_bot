from __future__ import annotations

from datetime import datetime, timezone
import queue
import time

import pandas as pd

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


def test_ws_tick_without_broker_timestamp_is_tagged_synthetic() -> None:
    mdm = _manager()

    normalized = mdm._normalize_ws_tick(
        {"instrument_token": TOKEN, "last_price": 100.0}
    )

    assert normalized is not None
    assert normalized["timestamp_quality"] == "synthetic"


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

    assert mdm._put_fallback_tick_nowait(
        {"symbol": SYMBOL, "instrument_token": TOKEN, "last_price": 101.0}
    ) is True
    assert mdm._put_fallback_tick_nowait(
        {"symbol": SYMBOL, "instrument_token": TOKEN, "last_price": 102.0}
    ) is True

    retained = mdm._fallback_tick_queue.get_nowait()
    assert retained["last_price"] == 102.0


def test_clock_flush_finalizes_idle_candle_without_next_tick() -> None:
    mdm = _manager()
    engine = mdm._get_engine(SYMBOL)
    candle_minute = pd.Timestamp.now(tz="Asia/Kolkata").floor("1min") - pd.Timedelta(minutes=2)
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
