from __future__ import annotations

import importlib

importlib.import_module("nifty_scalper_bot.data.quote_identity_extension")
from nifty_scalper_bot.data.data_hub import DataHub


class DummyMDM:
    def attach_tick_bus(self, _bus):  # noqa: ANN001
        return None


def test_quote_contract_adds_identity_and_tick_age():
    hub = DataHub(DummyMDM(), clock=lambda: 1000.0)
    try:
        hub.ingest_tick(
            {
                "symbol": "NIFTY24JAN100CE",
                "instrument_token": 12345,
                "last_price": 88.25,
                "timestamp": 999.0,
                "source": "ws",
            }
        )
        quote = hub.get_quote("NIFTY24JAN100CE", allow_pull=False)
        assert quote is not None
        assert quote["symbol"] == "NFO:NIFTY24JAN100CE"
        assert quote["tradingsymbol"] == "NFO:NIFTY24JAN100CE"
        assert quote["instrument_token"] == 12345
        assert quote["quote_update_version"] == 1
        assert quote["tick_age_ms"] == 1000.0
    finally:
        hub.close()
