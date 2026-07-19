from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.market_data_manager import MarketDataManager


def test_ws_tick_preserves_token_after_queue_processing() -> None:
    mdm = MarketDataManager(broker=None, websocket=None, settings={})
    mdm.register_symbol("NSE:NIFTY", 256265)

    mdm._process_queued_tick(
        {
            "instrument_token": 256265,
            "symbol": "NSE:NIFTY",
            "last_price": 24078.45,
            "exchange_timestamp": datetime(2026, 1, 1, tzinfo=timezone.utc),
        }
    )

    tick = mdm.get_latest_tick("NSE:NIFTY")
    assert tick is not None
    assert tick.get("instrument_token") == 256265
    assert tick.get("source") == "ws"
    assert tick.get("last_price") == 24078.45
    assert mdm.get_cached_ltp("NSE:NIFTY", require_ws=True) == 24078.45
    assert 256265 in mdm._confirmed_subscriptions or mdm._spot_ws_first_tick_seen
