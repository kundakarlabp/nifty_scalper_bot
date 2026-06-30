from __future__ import annotations

from collections import Counter
from typing import Any

from nifty_scalper_bot.data.data_hub import DataHub


class Store:
    def __init__(self):
        self.saved: list[tuple[str, dict[str, Any]]] = []
    def load_snapshot(self, kind):
        return None
    def save_snapshot(self, kind, payload):
        self.saved.append((kind, payload))


class MDM:
    def attach_tick_bus(self, bus):
        self.bus = bus


def _tick(i: int):
    return {
        "symbol": "NFO:NIFTY26JUN24000CE",
        "instrument_token": 123,
        "ltp": 100 + i,
        "last_price": 100 + i,
        "timestamp": 1_800_000_000 + i,
        "source": "ws",
    }


def test_datahub_ticks_do_not_checkpoint_full_state_per_tick():
    store = Store()
    hub = DataHub(MDM(), store=store)
    for i in range(10_000):
        hub.ingest_tick_sync(_tick(i))
    counts = Counter(kind for kind, _ in store.saved)
    assert counts["orders"] == 0
    assert counts["positions"] == 0
    assert counts["quotes"] < 10_000


def test_datahub_orders_and_positions_persist_immediately_and_shutdown_flushes_quotes():
    store = Store()
    hub = DataHub(MDM(), store=store)
    hub.ingest_tick_sync(_tick(1))
    hub.upsert_order({
        "order_id": "o1",
        "status": "OPEN",
        "symbol": "NFO:NIFTY26JUN24000CE",
    })
    hub.update_position({
        "symbol": "NFO:NIFTY26JUN24000CE",
        "quantity": 50,
        "average_price": 100,
    })
    counts = Counter(kind for kind, _ in store.saved)
    assert counts["orders"] >= 1
    assert counts["positions"] >= 1
    hub.close()
    latest = {kind: payload for kind, payload in store.saved}
    assert "NFO:NIFTY26JUN24000CE" in latest["quotes"]
    assert "o1" in latest["orders"]
    assert latest["positions"]
