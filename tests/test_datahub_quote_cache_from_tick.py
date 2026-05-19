from nifty_scalper_bot.data.data_hub import DataHub


class _MDM:
    def attach_tick_bus(self, _bus):
        return None


def test_datahub_ingest_tick_updates_quote_cache() -> None:
    hub = DataHub(_MDM())
    hub.ingest_tick({'symbol': 'NFO:NIFTY26MAY25000CE', 'ltp': 100.0, 'instrument_token': 123})
    q = hub.get_quote('NFO:NIFTY26MAY25000CE', allow_pull=False)
    assert q is not None
    assert float(q.get('ltp', 0.0)) == 100.0


def test_datahub_quote_update_version_increments() -> None:
    hub = DataHub(_MDM())
    hub.ingest_tick_sync({'symbol': 'NSE:NIFTY', 'ltp': 25000.0, 'timestamp': '2026-05-19T09:15:00+00:00'})
    assert hub.quote_update_version('NSE:NIFTY') == 1
    hub.ingest_tick_sync({'symbol': 'NSE:NIFTY', 'ltp': 25010.0, 'timestamp': '2026-05-19T09:15:01+00:00'})
    assert hub.quote_update_version('NSE:NIFTY') == 2
