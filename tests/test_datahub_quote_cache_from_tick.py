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
