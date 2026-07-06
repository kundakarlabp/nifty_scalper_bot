from __future__ import annotations

from datetime import datetime, timezone
import time

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
