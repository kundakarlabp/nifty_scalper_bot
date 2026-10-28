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


def test_hung_exit_with_unverified_tag_is_not_resubmitted(monkeypatch, tmp_path):
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
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(order_manager_core, "Thread", BlockedThread)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    lookups = []

    def unverified_exit(client_id):
        lookups.append(client_id)
        return {
            "order_id": "EXIT-1",
            "tag": "manual-square-off",
            "status": "OPEN",
            "tradingsymbol": "NIFTY2671423950CE",
            "transaction_type": "SELL",
            "quantity": 65,
            "product": "MIS",
        }

    monkeypatch.setattr(manager, "_find_open_order", unverified_exit)
    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )

    assert lookups
    assert order_id is None
    assert BlockedThread.starts == 1
    assert manager._is_uncertain_order(lookups[-1])

    later = order_manager_core.time.time() + 120
    monkeypatch.setattr(order_manager_core.time, "time", lambda: later)
    second = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
        tag="manual-square-off",
    )
    assert second is None
    assert BlockedThread.starts == 1


@pytest.mark.parametrize("second_tag", ["manual-square-off", "another-exit-tag"])
def test_hung_exit_hold_survives_restart_and_tag_change(
    monkeypatch, tmp_path, second_tag
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

    positions = PositionManager(str(tmp_path / "positions.json"))
    positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    first_manager = OrderManager(Broker(), positions, RateLimiter())
    monkeypatch.setattr(order_manager_core, "Thread", BlockedThread)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    monkeypatch.setattr(first_manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(first_manager, "_find_open_order", lambda _: None)
    kwargs = {
        "symbol": "NFO:NIFTY2671423950CE",
        "side": "SELL",
        "quantity": 65,
        "order_type": OrderType.MARKET,
        "check_risk": False,
        "intent": "EXIT",
        "tag": "manual-square-off",
    }
    assert first_manager.place_order(**kwargs) is None
    assert BlockedThread.starts == 1
    assert (tmp_path / "uncertain_exit_orders.json").exists()

    later = order_manager_core.time.time() + 7200
    monkeypatch.setattr(order_manager_core.time, "time", lambda: later)
    restarted = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    monkeypatch.setattr(restarted, "_lot_size_for_symbol", lambda _: 65)
    kwargs["tag"] = second_tag
    assert restarted.place_order(**kwargs) is None
    assert BlockedThread.starts == 1
    assert restarted._last_order_decision["block_reason"] == (
        "uncertain_exit_reconciliation_required"
    )
    health = restarted.execution_health_snapshot()
    assert health["uncertain_exit_orders_count"] == 1
    assert health["uncertain_exit_symbols"] == ["NFO:NIFTY2671423950CE"]
    assert (
        restarted.place_order(
            symbol="NFO:NIFTY2671424000PE",
            side="BUY",
            quantity=65,
            order_type=OrderType.LIMIT,
            price=100.0,
            stop_loss=95.0,
            take_profit=110.0,
            check_risk=False,
            intent="ENTRY",
            signal_id="new-entry-after-uncertain-exit",
        )
        is None
    )
    assert restarted._last_skip_reason == "uncertain_exit_reconciliation_required"
    assert BlockedThread.starts == 1


@pytest.mark.parametrize("broker_status", ["COMPLETE", "OPEN"])
def test_ambiguous_exit_cannot_adopt_unsettled_broker_order(
    monkeypatch, tmp_path, broker_status
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
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(order_manager_core, "Thread", BlockedThread)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(
        manager,
        "_find_open_order",
        lambda client_id: {
            "order_id": "EXIT-LATE",
            "client_order_id": client_id,
            "tradingsymbol": "NIFTY2671423950CE",
            "transaction_type": "SELL",
            "quantity": 65,
            "product": "MIS",
            "status": broker_status,
            "filled_quantity": 32 if broker_status == "OPEN" else 65,
        },
    )
    assert (
        manager.place_order(
            symbol="NFO:NIFTY2671423950CE",
            side="SELL",
            quantity=65,
            order_type=OrderType.MARKET,
            check_risk=False,
            intent="EXIT",
        )
        is None
    )
    assert BlockedThread.starts == 1
    assert manager._uncertain_exit_ledger.get("NFO:NIFTY2671423950CE")
    assert not manager._orders


def test_exit_submission_exception_not_retried(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker:
        is_simulated_adapter = True

    manager = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(order_manager_core.time, "sleep", lambda _: None)
    calls = []

    def unavailable(_kwargs):
        calls.append(1)
        raise TimeoutError("response timed out")

    monkeypatch.setattr(manager, "_submit_broker_order", unavailable)
    assert (
        manager.place_order(
            symbol="NFO:NIFTY2671423950CE",
            side="SELL",
            quantity=65,
            order_type=OrderType.MARKET,
            check_risk=False,
            intent="EXIT",
        )
        is None
    )
    assert len(calls) == 1
    assert manager._uncertain_exit_ledger.get("NFO:NIFTY2671423950CE")


def test_corrupt_exit_write_ahead_state_blocks_orders(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    (tmp_path / "uncertain_exit_orders.json").write_text("not-json")

    class Broker:
        is_simulated_adapter = True

    manager = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    assert (
        manager.place_order(
            symbol="NFO:NIFTY2671423950CE",
            side="SELL",
            quantity=65,
            order_type=OrderType.MARKET,
            check_risk=False,
            intent="EXIT",
        )
        is None
    )
    assert manager._last_order_decision["block_reason"] == (
        "uncertain_exit_ledger_unreadable"
    )


def test_acknowledged_exit_releases_durable_submission_hold(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker:
        is_simulated_adapter = True

    manager = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(
        manager, "_submit_broker_order", lambda _: {"order_id": "EXIT-CONFIRMED"}
    )
    order_id = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
    )
    assert order_id == "EXIT-CONFIRMED"
    assert manager.execution_health_snapshot()["uncertain_exit_orders_count"] == 0


def test_failed_exit_hold_clear_keeps_subsequent_orders_blocked(monkeypatch, tmp_path):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")
    monkeypatch.setenv("DATA_DIR", str(tmp_path))

    class Broker:
        is_simulated_adapter = True

    manager = OrderManager(
        Broker(), PositionManager(str(tmp_path / "positions.json")), RateLimiter()
    )
    manager._positions.open_position(
        "NFO:NIFTY2671423950CE", "LONG", 65, 100.0, order_id="entry-1"
    )
    monkeypatch.setattr(manager, "_lot_size_for_symbol", lambda _: 65)
    monkeypatch.setattr(
        manager, "_submit_broker_order", lambda _: {"order_id": "EXIT-CONFIRMED"}
    )
    original_flush = manager._uncertain_exit_ledger._flush

    def fail_clear():
        if not manager._uncertain_exit_ledger.snapshot():
            raise OSError("simulated disk failure")
        original_flush()

    monkeypatch.setattr(manager._uncertain_exit_ledger, "_flush", fail_clear)
    exit_order = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="SELL",
        quantity=65,
        order_type=OrderType.MARKET,
        check_risk=False,
        intent="EXIT",
    )
    assert exit_order == "EXIT-CONFIRMED"
    assert manager.execution_health_snapshot()["uncertain_exit_orders_count"] == 1
    entry_order = manager.place_order(
        symbol="NFO:NIFTY2671423950CE",
        side="BUY",
        quantity=65,
        order_type=OrderType.LIMIT,
        price=100.0,
        stop_loss=95.0,
        take_profit=110.0,
        check_risk=False,
        intent="ENTRY",
        signal_id="after-failed-clear",
    )
    assert entry_order is None
    assert manager._last_skip_reason == "uncertain_exit_reconciliation_required"
