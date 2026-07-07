from __future__ import annotations

import importlib
from types import SimpleNamespace

importlib.import_module("nifty_scalper_bot.data.quote_identity_extension")
from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.quote_identity_extension import stamp_quote_identity


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
                "source": "ws",
            }
        )
        quote = hub.get_quote("NIFTY24JAN100CE", allow_pull=False)
        assert quote is not None
        assert quote["symbol"] == "NFO:NIFTY24JAN100CE"
        assert quote["tradingsymbol"] == "NFO:NIFTY24JAN100CE"
        assert quote["instrument_token"] == 12345
        assert quote["quote_update_version"] == 1
        assert quote["quote_identity_timestamp_source"] == "hub_clock_fallback"
        assert quote["tick_age_ms"] == 0.0
        assert quote["quote_age_s"] == 0.0
    finally:
        hub.close()


def test_quote_identity_falls_back_to_arrival_when_timestamp_is_future():
    now_epoch = 1_783_415_880.0  # 2026-07-07 14:48 IST
    hub = SimpleNamespace(
        _now=lambda: now_epoch,
        _canonical_quote_symbol=lambda symbol: f"NFO:{symbol}" if ":" not in str(symbol) else str(symbol),
        quote_update_version=lambda _symbol: 7,
    )
    quote = {
        "symbol": "NIFTY24JAN100CE",
        "instrument_token": 12345,
        "last_price": 88.25,
        "timestamp": "2026-07-07T20:18:01+05:30",
        "received_at": now_epoch,
    }

    stamped = stamp_quote_identity(hub, "NIFTY24JAN100CE", quote)

    assert stamped["quote_update_version"] == 7
    assert stamped["quote_identity_timestamp_source"] == "received_at_for_timestamp_future_guard"
    assert stamped["tick_age_ms"] == 0.0
    assert stamped["quote_age_s"] == 0.0
