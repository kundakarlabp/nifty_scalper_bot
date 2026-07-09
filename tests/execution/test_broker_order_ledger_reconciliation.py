from __future__ import annotations

import json

from nifty_scalper_bot.execution.position_manager import PositionManager


SYMBOL = "NFO:NIFTY2671423950CE"
BROKER_SYMBOL = "NIFTY2671423950CE"


class Broker:
    def __init__(self, *, positions=None, orders=None, exc: Exception | None = None) -> None:
        self.positions = {"net": []} if positions is None else positions
        self.orders = [] if orders is None else orders
        self.exc = exc
        self.position_calls = 0
        self.order_calls = 0

    def get_positions(self):
        self.position_calls += 1
        if self.exc is not None:
            raise self.exc
        return self.positions

    def get_orders(self):
        self.order_calls += 1
        return self.orders


def _filled_order(order_id: str = "broker-filled") -> dict[str, object]:
    return {
        "order_id": order_id,
        "tradingsymbol": BROKER_SYMBOL,
        "transaction_type": "BUY",
        "quantity": 65,
        "filled_quantity": 65,
        "average_price": 94.75,
        "status": "COMPLETE",
        "product": "MIS",
        "order_timestamp": "2026-07-09 09:31:01",
    }


def _open_order(order_id: str = "broker-open") -> dict[str, object]:
    return {
        "order_id": order_id,
        "tradingsymbol": BROKER_SYMBOL,
        "transaction_type": "BUY",
        "quantity": 65,
        "filled_quantity": 0,
        "average_price": 0,
        "status": "OPEN",
        "product": "MIS",
        "order_timestamp": "2026-07-09 09:31:01",
    }


def test_unknown_filled_broker_order_with_flat_position_is_persistently_resolved(tmp_path):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_file))
    manager.set_broker_client(Broker(positions={"net": []}))

    manager.apply_broker_order_update("broker-filled", _filled_order())

    ledger = manager.get_broker_order_ledger()
    assert ledger["broker-filled"]["classification"] == "resolved_external_flat"
    assert manager.get_quarantined_broker_exposures() == {}
    assert manager.current_entry_protection_blocker(SYMBOL) is None

    payload = json.loads(state_file.read_text(encoding="utf-8"))
    assert payload["broker_order_ledger"]["broker-filled"]["classification"] == "resolved_external_flat"

    restored = PositionManager(state_file=str(state_file))
    assert restored.get_broker_order_ledger()["broker-filled"]["classification"] == "resolved_external_flat"
    assert restored.current_entry_protection_blocker(SYMBOL) is None


def test_unknown_filled_broker_order_with_open_position_is_quarantined_once(tmp_path, caplog):
    state_file = tmp_path / "positions.json"
    manager = PositionManager(state_file=str(state_file))
    manager.set_broker_client(
        Broker(
            positions={
                "net": [
                    {
                        "tradingsymbol": BROKER_SYMBOL,
                        "quantity": 65,
                        "average_price": 94.75,
                    }
                ]
            }
        )
    )

    manager.apply_broker_order_update("broker-filled", _filled_order())
    manager.apply_broker_order_update("broker-filled", _filled_order())

    ledger = manager.get_broker_order_ledger()["broker-filled"]
    assert ledger["classification"] == "broker_position_quarantined"
    assert ledger["broker_position_state"] == "open"
    assert manager.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"
    exposures = manager.get_quarantined_broker_exposures()
    assert exposures[SYMBOL]["order_id"] == "broker-filled"
    assert exposures[SYMBOL]["source"] == "broker_order_ledger"
    assert caplog.text.count("BROKER_EXTERNAL_ORDER_QUARANTINED") <= 1

    restored = PositionManager(state_file=str(state_file))
    assert restored.get_broker_order_ledger()["broker-filled"]["classification"] == "broker_position_quarantined"
    assert restored.current_entry_protection_blocker(SYMBOL) == "broker_exposure_quarantined"


def test_unknown_open_broker_order_blocks_until_resolved(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.apply_broker_order_update("broker-open", _open_order())

    ledger = manager.get_broker_order_ledger()["broker-open"]
    assert ledger["classification"] == "active_external_order"
    assert manager.current_entry_protection_blocker(SYMBOL) == "active_external_order"

    manager.apply_broker_order_update(
        "broker-open",
        {
            **_open_order(),
            "status": "CANCELLED",
            "filled_quantity": 0,
        },
    )

    assert manager.get_broker_order_ledger()["broker-open"]["classification"] == "resolved_external_terminal"
    assert manager.current_entry_protection_blocker(SYMBOL) is None


def test_reconcile_broker_orders_is_deterministic_and_canonicalizes_symbols(tmp_path):
    manager = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager.set_broker_client(
        Broker(
            positions={"net": []},
            orders=[
                {**_open_order("later"), "order_timestamp": "2026-07-09 09:31:02"},
                {**_filled_order("earlier"), "order_timestamp": "2026-07-09 09:31:01"},
            ],
        )
    )

    counts = manager.reconcile_broker_orders()

    assert counts == {"seen": 2, "managed": 0, "external": 1, "resolved": 1}
    ledger = manager.get_broker_order_ledger()
    assert ledger["earlier"]["symbol"] == SYMBOL
    assert ledger["earlier"]["classification"] == "resolved_external_flat"
    assert ledger["later"]["classification"] == "active_external_order"
