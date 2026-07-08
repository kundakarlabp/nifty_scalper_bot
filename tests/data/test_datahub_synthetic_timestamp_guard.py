from __future__ import annotations

import pandas as pd

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.execution.readiness import evaluate_quote_readiness


class _MdmNoop:
    def attach_tick_bus(self, _tick_bus) -> None:
        return None


def _quote_payload(**overrides):
    payload = {
        "instrument_token": 12345,
        "ltp": 100.0,
        "bid": 99.5,
        "ask": 100.5,
        "depth_available": True,
    }
    payload.update(overrides)
    return payload


def test_store_quote_without_timestamp_is_tagged_synthetic_and_not_hard_ready() -> None:
    hub = DataHub(_MdmNoop())
    symbol = "NFO:NIFTY26JUL24000CE"

    hub.store_quote(symbol, _quote_payload(), source="ws")
    quote = hub.get_quote(symbol, allow_pull=False)
    readiness = evaluate_quote_readiness(symbol, quote, require_fresh=True, max_age_s=2.0)

    assert quote is not None
    assert quote["timestamp_quality"] == "synthetic"
    assert quote["synthetic_timestamp"] is True
    assert quote["hard_readiness_eligible"] is False
    assert quote["tradable_quote"] is False
    assert readiness.tradable_quote_ready is False
    assert readiness.reason == "timestamp_quality_unusable"
    assert hub.get_cached_ltp(symbol, max_age_seconds=2.0) is None


def test_invalid_timestamp_is_tagged_invalid_and_rejected_for_guarded_cached_ltp() -> None:
    hub = DataHub(_MdmNoop())
    symbol = "NFO:NIFTY26JUL24000PE"

    hub.ingest_tick_sync(_quote_payload(symbol=symbol, timestamp="not-a-date", source="ws"))
    quote = hub.get_quote(symbol, allow_pull=False)

    assert quote is not None
    assert quote["timestamp_quality"] == "invalid"
    assert quote["hard_readiness_eligible"] is False
    assert hub.get_cached_ltp(symbol, max_age_seconds=2.0) is None
    assert hub.get_cached_ltp(symbol) == 100.0


def test_valid_broker_timestamp_can_satisfy_guarded_cached_ltp() -> None:
    hub = DataHub(_MdmNoop())
    symbol = "NSE:NIFTY"
    now = pd.Timestamp.utcnow().isoformat()

    hub.ingest_tick_sync(_quote_payload(symbol=symbol, timestamp=now, source="ws"))
    quote = hub.get_quote(symbol, allow_pull=False)

    assert quote is not None
    assert quote["timestamp_quality"] == "broker"
    assert quote["hard_readiness_eligible"] is True
    assert hub.get_cached_ltp(symbol, max_age_seconds=2.0, require_ws=True) == 100.0


def test_received_at_quality_preserved_when_it_is_the_only_valid_time_proof() -> None:
    hub = DataHub(_MdmNoop())
    symbol = "NSE:NIFTY"

    hub.ingest_tick_sync(_quote_payload(symbol=symbol, received_at=pd.Timestamp.utcnow().timestamp(), source="ws"))
    quote = hub.get_quote(symbol, allow_pull=False)

    assert quote is not None
    assert quote["timestamp_quality"] == "received_at"
    assert quote["hard_readiness_eligible"] is True
