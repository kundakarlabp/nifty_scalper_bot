from __future__ import annotations

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
