"""Broker-free regression for a submitted order with a delayed response."""

from nifty_scalper_bot.execution import order_manager_core
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


def test_hung_submission_keeps_first_reconciled_order(monkeypatch, tmp_path):
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

    def find_order(_client_id):
        lookups.append(1)
        if len(lookups) == 1:
            return {"order_id": "BROKER-1", "status": "OPEN"}
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

    assert order_id == "BROKER-1"
    assert BlockedThread.starts == 1
