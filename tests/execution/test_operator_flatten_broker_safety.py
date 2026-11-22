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
        self._orders = {"stop-1": {"order_id": "stop-1"}}
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


def test_unowned_same_symbol_nifty_stop_is_not_cancelled_or_flattened() -> None:
    class MixedBroker(_Broker):
        def get_orders(self):
            return super().get_orders() + [
                {
                    "order_id": "manual-nifty-stop",
                    "tradingsymbol": SYMBOL,
                    "status": "TRIGGER PENDING",
                    "transaction_type": "SELL",
                }
            ]

    broker = MixedBroker()
    manager = _Manager(broker)
    result = flatten_all(manager)

    assert broker.cancel_calls == []
    assert manager.sent == []
    assert result["failed"][0]["reason"] == "broker_proof_unavailable:ValueError"


def test_cancel_pending_orders_keeps_untracked_same_symbol_order() -> None:
    from nifty_scalper_bot.execution.operator_control import cancel_pending_orders

    class MixedBroker(_Broker):
        def get_orders(self):
            return super().get_orders() + [
                {
                    "order_id": "manual-nifty-stop",
                    "tradingsymbol": SYMBOL,
                    "status": "TRIGGER PENDING",
                }
            ]

    broker = MixedBroker()
    manager = _Manager(broker)
    outcome = cancel_pending_orders(manager)
    assert outcome["cancelled"] == ["stop-1"]
    assert broker.cancel_calls == ["stop-1"]


def test_emergency_preserves_unrelated_account_stops() -> None:
    class MixedBroker(_Broker):
        def get_orders(self):
            return super().get_orders() + [
                {
                    "order_id": "unrelated-equity-stop",
                    "tradingsymbol": "RELIANCE",
                    "status": "TRIGGER PENDING",
                    "transaction_type": "SELL",
                },
                {
                    "order_id": "unrelated-banknifty-stop",
                    "tradingsymbol": "BANKNIFTY26NOV45000CE",
                    "status": "TRIGGER PENDING",
                    "transaction_type": "SELL",
                },
            ]

    broker = MixedBroker()
    manager = _Manager(broker)

    result = emergency_stop(manager, reason="nifty-only")

    assert result["flatten"]["failed"] == []
    assert broker.cancel_calls == ["stop-1"]
    assert len(manager.sent) == 1
    assert manager.sent[0]["symbol"] == SYMBOL


def test_no_nifty_position_cannot_disarm_other_instrument() -> None:
    class OtherOnlyBroker(_Broker):
        def get_positions(self):
            return {
                "net": [
                    {
                        "tradingsymbol": "RELIANCE",
                        "quantity": 100,
                        "product": "MIS",
                    }
                ]
            }

        def get_orders(self):
            return [
                {
                    "order_id": "equity-stop",
                    "tradingsymbol": "RELIANCE",
                    "status": "TRIGGER PENDING",
                }
            ]

    broker = OtherOnlyBroker()
    manager = _Manager(broker)
    result = emergency_stop(manager, reason="nifty-only")

    assert result["cancel"]["cancelled"] == []
    assert result["flatten"]["submitted"] == []
    assert broker.cancel_calls == []
    assert manager.sent == []


def test_post_cancel_readback_outage_reports_unverified_protection(caplog) -> None:
    class ReadbackOutageBroker(_Broker):
        def __init__(self):
            super().__init__()
            self.reads = 0

        def get_orders(self):
            self.reads += 1
            if self.reads == 3:
                raise RuntimeError("broker orders unavailable after cancel")
            return super().get_orders()

    broker = ReadbackOutageBroker()
    manager = _Manager(broker)

    with caplog.at_level(logging.CRITICAL):
        result = flatten_all(manager)

    assert result["cancel"]["cancelled"] == ["stop-1"]
    assert result["failed"]
    assert manager.sent == []
    assert "OPERATOR_FLATTEN_PROTECTION_UNVERIFIED" in caplog.text


def test_multi_product_nifty_positions_keep_both_exchange_stops() -> None:
    class TwoProductBroker(_Broker):
        def get_positions(self):
            return {
                "net": [
                    {"tradingsymbol": SYMBOL, "quantity": 65, "product": "MIS"},
                    {"tradingsymbol": SYMBOL, "quantity": -65, "product": "NRML"},
                ]
            }

        def get_orders(self):
            return [
                {
                    "order_id": "mis-stop",
                    "tradingsymbol": SYMBOL,
                    "status": "TRIGGER PENDING",
                    "product": "MIS",
                },
                {
                    "order_id": "nrml-stop",
                    "tradingsymbol": SYMBOL,
                    "status": "TRIGGER PENDING",
                    "product": "NRML",
                },
            ]

    broker = TwoProductBroker()
    manager = _Manager(broker)
    result = flatten_all(manager)

    assert result["failed"]
    assert result["failed"][0]["reason"] == "broker_proof_unavailable:ValueError"
    assert not broker.cancel_calls
    assert not manager.sent
