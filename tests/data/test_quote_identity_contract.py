from __future__ import annotations

from nifty_scalper_bot.data.data_hub import DataHub


class _DummyMDM:
    def attach_tick_bus(self, _bus):  # noqa: ANN001
        return None


def test_datahub_stamps_quote_identity_and_freshness_metadata():
    hub = DataHub(_DummyMDM(), clock=lambda: 1000.0)
    try:
        hub.ingest_tick(
            {
                "symbol": "NIFTY2670724250PE",
                "instrument_token": 12345,
                "last_price": 88.25,
                "bid": 88.0,
                "ask": 88.5,
                "timestamp": 999.0,
                "source": "ws",
            }
        )

        quote = hub.get_quote("NFO:NIFTY2670724250PE", allow_pull=False)

        assert quote is not None
        assert quote["symbol"] == "NFO:NIFTY2670724250PE"
        assert quote["tradingsymbol"] == "NFO:NIFTY2670724250PE"
        assert quote["trading_symbol"] == "NFO:NIFTY2670724250PE"
        assert quote["instrument_symbol"] == "NFO:NIFTY2670724250PE"
        assert quote["exchange_symbol"] == "NFO:NIFTY2670724250PE"
        assert quote["instrument_token"] == 12345
        assert quote["quote_update_version"] == 1
        assert quote["last_tick_ts_ms"] == 999000.0
        assert quote["tick_age_ms"] == 1000.0
        assert quote["quote_identity_source"] == "datahub_quote_contract"
    finally:
        hub.close()


def test_datahub_get_tick_by_token_returns_identity_stamped_quote():
    hub = DataHub(_DummyMDM(), clock=lambda: 1000.0)
    try:
        hub.ingest_tick(
            {
                "symbol": "NIFTY2670724250PE",
                "instrument_token": 12345,
                "last_price": 88.25,
                "timestamp": 999.0,
                "source": "ws",
            }
        )

        quote = hub.get_tick_by_token(12345)

        assert quote is not None
        assert quote["symbol"] == "NFO:NIFTY2670724250PE"
        assert quote["tradingsymbol"] == "NFO:NIFTY2670724250PE"
        assert quote["quote_update_version"] == 1
        assert quote["tick_age_ms"] == 1000.0
    finally:
        hub.close()
