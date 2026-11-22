"""Broker-free operator emergency flatten must not outrun cancellation truth."""

from __future__ import annotations

import logging

import pytest

from nifty_scalper_bot.execution.operator_control import (
    _cancel_order,
    emergency_stop,
    flatten_all,
)

SYMBOL = "NFO:NIFTY26NOV25700CE"


class _Broker:
    def __init__(self, *, quantity: int = 65, product: str = "MIS") -> None:
        self.quantity = quantity
        self.product = product
        self.status = "TRIGGER PENDING"
        self.cancel_response = True
        self.cancel_transitions = True
        self.invalid_snapshot = False
        self.orders_unavailable = False
        self.cancel_calls: list[str] = []

    def get_positions(self):
        if self.invalid_snapshot:
            return None
        return {
            "net": [
                {
                    "tradingsymbol": SYMBOL,
                    "quantity": self.quantity,
                    "product": self.product,
                }
            ]
        }

    def get_orders(self):
        if self.orders_unavailable:
            return None
        return [
            {
                "order_id": "stop-1",
                "tradingsymbol": SYMBOL,
                "status": self.status,
                "transaction_type": "SELL",
            }
        ]

    def cancel_order(self, order_id: str):
        self.cancel_calls.append(order_id)
        if not self.cancel_response:
            return False
        if self.cancel_transitions:
            self.status = "CANCELLED"
        return {"order_id": order_id}


class _Manager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self._logger = logging.getLogger(__name__)
        self.sent: list[dict] = []

    def is_live_mode(self) -> bool:
        return True

    def _lot_size_for_symbol(self, _symbol: str) -> int:
        return 65

    def cancel_order(self, order_id: str):
        return self._broker.cancel_order(order_id)

    def place_order(self, **kwargs):
        self.sent.append(dict(kwargs))
        return f"close-{len(self.sent)}"


def test_cancellation_returning_false_is_not_success() -> None:
    broker = _Broker()
    broker.cancel_response = False
    manager = _Manager(broker)

    assert _cancel_order(manager, "stop-1") is False


def test_emergency_does_not_oversell_when_stop_cancellation_unconfirmed() -> None:
    broker = _Broker()
    broker.cancel_transitions = False
    manager = _Manager(broker)

    result = emergency_stop(manager, reason="test-unconfirmed-stop")

    assert not manager.sent
    assert result["flatten"]["failed"]
    assert broker.status == "TRIGGER PENDING"


def test_emergency_reconciles_cancellation_and_flattens_one_lot() -> None:
    broker = _Broker()
    manager = _Manager(broker)

    result = emergency_stop(manager, reason="test-verified-stop")

    assert result["cancel"]["cancelled"] == ["stop-1"]
    assert result["flatten"]["failed"] == []
    assert len(manager.sent) == 1
    assert manager.sent[0]["quantity"] == 65
    assert manager.sent[0]["product"] == "MIS"
    assert manager.sent[0]["side"] == "SELL"
    assert manager.sent[0]["intent"] == "REDUCE"


def test_live_operator_cancellation_uses_broker_orderbook_not_local_snapshot() -> None:
    broker = _Broker()
    manager = _Manager(broker)
    manager.get_open_orders = lambda: [
        {"order_id": "unrelated-local-ghost", "status": "OPEN"}
    ]

    result = emergency_stop(manager, reason="verify-exchange-stop-cancel")

    assert broker.cancel_calls == ["stop-1"]
    assert result["flatten"]["failed"] == []
    assert len(manager.sent) == 1


def test_emergency_preserves_verified_nrml_position_product() -> None:
    broker = _Broker(product="NRML")
    manager = _Manager(broker)

    result = emergency_stop(manager)

    assert not result["flatten"]["failed"]
    assert manager.sent[0]["product"] == "NRML"


@pytest.mark.parametrize("bad_quantity", [35, -35, 66])
def test_emergency_never_submits_fractional_nifty_lot(
    bad_quantity: int,
) -> None:
    broker = _Broker(quantity=bad_quantity)
    manager = _Manager(broker)

    result = flatten_all(manager)

    assert not manager.sent
    assert result["failed"]
    assert not broker.cancel_calls, "invalid exposure cannot disarm resting stop"


def test_unavailable_broker_positions_do_not_fall_back_to_local_for_live() -> None:
    broker = _Broker()
    broker.invalid_snapshot = True
    manager = _Manager(broker)

    result = flatten_all(manager)

    assert not manager.sent
    assert result["failed"]
    assert not broker.cancel_calls


def test_broker_orderbook_unavailable_keeps_existing_protection() -> None:
    broker = _Broker()
    broker.orders_unavailable = True
    manager = _Manager(broker)

    result = flatten_all(manager)

    assert not manager.sent
    assert result["failed"]
    assert not broker.cancel_calls


def test_confirmed_stop_fill_cannot_cause_second_exit() -> None:
    broker = _Broker()
    manager = _Manager(broker)

    def cancellation_fills(_order_id):
        broker.quantity = 0
        broker.status = "COMPLETE"
        return {"order_id": "stop-1"}

    manager.cancel_order = cancellation_fills
    result = flatten_all(manager)

    assert not manager.sent
    assert result["submitted"] == []
