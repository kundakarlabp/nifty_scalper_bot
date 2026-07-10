from __future__ import annotations

import time

from nifty_scalper_bot.data.data_hub import DataHub


class _Mdm:
    def attach_tick_bus(self, _tick_bus) -> None:
        return None


def test_poll_source_does_not_update_ws_arrival_but_stream_does() -> None:
    hub = DataHub(market_data_manager=_Mdm())
    poll_tick = {
        'symbol': 'NSE:NIFTY',
        'instrument_token': 256265,
        'ltp': 24000.0,
        'timestamp': 1_000_000,
        'source': 'poll',
    }
    hub.ingest_tick_sync(poll_tick)
    assert hub._last_ws_arrival.get('NSE:NIFTY', 0.0) == 0.0

    stream_tick = dict(poll_tick)
    stream_tick['timestamp'] = 1_100_000
    stream_tick['source'] = 'stream'
    hub.ingest_tick_sync(stream_tick)
    assert hub._last_ws_arrival.get('NSE:NIFTY', 0.0) > 0.0


def test_historical_source_quote_still_uses_age_for_freshness() -> None:
    now = time.time()
    hub = DataHub(market_data_manager=_Mdm(), clock=lambda: now)
    hub._quotes["NSE:NIFTY"] = {
        "symbol": "NSE:NIFTY",
        "ltp": 24000.0,
        "timestamp": now - 120.0,
        "source": "historical",
    }
    hub._start_mono = hub._monotonic() - hub._warmup_grace_s - 1.0

    fresh, meta = hub.is_fresh("NSE:NIFTY", threshold_ms=60_000)

    assert fresh is False
    assert meta["source"] == "historical"
    assert meta["reason"] == "stale"
