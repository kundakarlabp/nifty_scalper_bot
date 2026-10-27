"""Broker-free regression for a submitted order with a delayed response."""

import pytest

from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


@pytest.mark.parametrize("identity_field", ["client_order_id", "tag"])
def test_hung_submission_keeps_first_reconciled_order(
    monkeypatch, tmp_path, identity_field
):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker:
        is_simulated_adapter = True

    class BlockedThread:
        starts = 0

        def __init__(self, **kwargs):
            pass

        def start(self):
            type(self).starts += 1

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    broker = Broker()
    manager = OrderManager(
        broker, PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    monkeypatch.setattr(order_manager_core, "Thread", BlockedThread)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(manager, "_confirm_fill_fast", lambda *args, **kwargs: False)
    lookups = []

    def find_order(client_id):
        lookups.append(1)
        if len(lookups) == 1:
            order = {
                "order_id": "BROKER-1",
                "status": "OPEN",
                "tradingsymbol": "NIFTY2671423950CE",
                "transaction_type": "BUY",
                "quantity": 65,
                "product": "MIS",
            }
            order[identity_field] = client_id
            return order
        return None

    monkeypatch.setattr(manager, "_find_open_order", find_order)
    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id="hung-order-test",
    )

    assert BlockedThread.starts >= 1
    assert lookups
    assert order_id == "BROKER-1"
    assert BlockedThread.starts == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("tradingsymbol", "NIFTY2671423950PE"),
        ("transaction_type", "SELL"),
        ("quantity", 130),
        ("product", "CNC"),
        ("client_order_id", "wrong-id"),
        ("client_order_id", None),
        ("quantity", None),
    ],
)
def test_hung_submission_does_not_adopt_unverified_order(
    monkeypatch, tmp_path, field, value
):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker:
        is_simulated_adapter = True

    class BlockedThread:
        starts = 0

        def __init__(self, **kwargs):
            pass

        def start(self):
            type(self).starts += 1

        def join(self, timeout=None):
            pass

        def is_alive(self):
            return True

    manager = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    monkeypatch.setattr(order_manager_core, "Thread", BlockedThread)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(manager, "_confirm_fill_fast", lambda *args, **kwargs: False)
    lookups = []

    def wrong_contract(client_id):
        lookups.append(client_id)
        order = {
            "order_id": "OTHER-1",
            "status": "OPEN",
            "client_order_id": client_id,
            "tradingsymbol": "NIFTY2671423950CE",
            "transaction_type": "BUY",
            "quantity": 65,
            "product": "MIS",
        }
        order[field] = value
        return order

    monkeypatch.setattr(manager, "_find_open_order", wrong_contract)
    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id="wrong-contract-test",
    )

    assert lookups
    assert order_id is None
    assert BlockedThread.starts == 1
