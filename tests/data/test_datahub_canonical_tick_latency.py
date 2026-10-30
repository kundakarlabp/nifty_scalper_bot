"""Canonical MDM ticks avoid duplicate parsing without weakening tick identity."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.data.data_hub import DataHub
from nifty_scalper_bot.data.market_data_manager import MarketDataManager

SYMBOL = "NFO:NIFTY26NOV23000CE"


def _canonical_tick() -> dict:
    now = datetime.now(timezone.utc)
    return {
        "symbol": SYMBOL,
        "instrument_token": 13800000,
        "ltp": 100.0,
        "last_price": 100.0,
        "timestamp": now.isoformat(),
        "timestamp_ms": now.timestamp() * 1000,
        "timestamp_source": "exchange_timestamp",
        "timestamp_quality": "exchange",
        "source_timestamp_valid": True,
        "received_at": now.timestamp(),
        "source": "ws",
        "depth_available": True,
        "tradable_quote": True,
        "bid": 99.9,
        "ask": 100.1,
    }


def test_canonical_mdm_tick_uses_validated_millisecond_timestamp(monkeypatch) -> None:
    hub = DataHub(MarketDataManager(kite=None))
    payload = _canonical_tick()
    seen: list[dict] = []
    hub.subscribe_ticks(SYMBOL, seen.append, token=13800000)

    def forbid_reparse(_value):
        raise AssertionError("MDM normalized this timestamp already")

    monkeypatch.setattr(hub, "_coerce_tick_timestamp", forbid_reparse)
    hub.ingest_tick_sync(payload)

    assert len(seen) == 1
    assert seen[0]["timestamp_ms"] == payload["timestamp_ms"]
    assert seen[0]["timestamp"] == payload["timestamp"]
    assert seen[0]["ltp"] == 100.0
    assert seen[0]["bid"] == 99.9
    assert seen[0]["ask"] == 100.1


def test_noncanonical_tick_still_receives_canonical_timestamp(monkeypatch) -> None:
    hub = DataHub(MarketDataManager(kite=None))
    payload = _canonical_tick()
    payload.pop("source_timestamp_valid")
    calls: list[object] = []
    original = hub._coerce_tick_timestamp

    def track(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(hub, "_coerce_tick_timestamp", track)
    hub.ingest_tick_sync(payload)
    assert calls
    latest = hub.get_quote(SYMBOL, allow_pull=False)
    assert latest is not None
    assert latest["timestamp_ms"] > 0
