from __future__ import annotations

import time

import pytest

from nifty_scalper_bot.config.base import RiskConfig
from nifty_scalper_bot.execution.options_policy import OptionsExecutionPolicy
from nifty_scalper_bot.execution.order_executor import OrderExecutor
from nifty_scalper_bot.utils.errors import BrokerError, OrderPlacementError


def test_round_to_tick_snapshots_to_nearest_tick() -> None:
    policy = OptionsExecutionPolicy()
    assert policy.round_to_tick(123.72) == pytest.approx(123.7)


def test_validate_qty_enforces_lot_size() -> None:
    policy = OptionsExecutionPolicy()
    policy.set_lot_size_lookup(lambda symbol: 25 if symbol else 0)
    with pytest.raises(OrderPlacementError):
        policy.validate_qty("NIFTY", 10)
    policy.validate_qty("NIFTY", 50)


def test_price_guard_rejects_stale_quotes() -> None:
    policy = OptionsExecutionPolicy()
    now_ns = time.time_ns()
    with pytest.raises(OrderPlacementError):
        policy.price_guard(
            100.0, 101.0, now_ns - (policy.max_tick_age_ms + 10) * 1_000_000
        )


class StaticMDM:
    def __init__(self) -> None:
        self.quotes: dict[str, dict[str, float]] = {}

    def set_quote(self, symbol: str, bid: float, ask: float) -> None:
        self.quotes[symbol] = {
            "bid": bid,
            "ask": ask,
            "ts_ns": time.time_ns(),
        }

    def get_last_quote(self, symbol: str) -> dict[str, float]:
        return self.quotes[symbol]

    def lot_size_for_symbol(self, symbol: str) -> int:
        return 25

    def now_ns(self) -> int:
        return time.time_ns()


class FlakyBroker:
    def __init__(self) -> None:
        self.calls = 0
        self.orders: dict[str, dict[str, object]] = {}
        self.status: dict[str, dict[str, object]] = {}

    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        self.calls += 1
        cid = str(payload["client_order_id"])
        if self.calls == 1:
            self.orders[cid] = {"order_id": "O123"}
            self.status["O123"] = {"filled_qty": payload["qty"]}
            raise BrokerError("network flap")
        return self.orders[cid]

    def get_order_by_client_order_id(self, cid: str) -> dict[str, object] | None:
        return self.orders.get(cid)

    def get_order_status(self, order_id: str) -> dict[str, object]:
        return self.status[order_id]

    def cancel_order(self, order_id: str) -> None:
        raise AssertionError("cancel should not be invoked in idempotent path")


def test_order_executor_idempotent_retry_recovers_order() -> None:
    risk = RiskConfig(
        max_daily_trades=10, max_order_notional=2_500_000, allow_short=True
    )
    broker = FlakyBroker()
    mdm = StaticMDM()
    mdm.set_quote("NIFTY", 100.0, 101.0)
    executor = OrderExecutor(broker, risk, mdm)
    order_id = executor.place_market_order("NIFTY", "BUY", 25, 100.0)
    assert order_id == "O123"
    assert broker.calls == 1


def test_reconcile_open_orders_uses_client_order_identifier() -> None:
    risk = RiskConfig(
        max_daily_trades=10, max_order_notional=2_500_000, allow_short=True
    )
    broker = FlakyBroker()
    mdm = StaticMDM()
    mdm.set_quote("NIFTY", 100.0, 101.0)
    executor = OrderExecutor(broker, risk, mdm)
    executor.place_market_order("NIFTY", "BUY", 25, 100.0, client_order_id="CID123")
    snapshot = executor.reconcile_open_orders()
    assert "CID123" in snapshot
    assert snapshot["CID123"]["order_id"] == "O123"


class RejectingBroker:
    def __init__(self) -> None:
        self.orders: dict[str, dict[str, object]] = {}
        self.cancelled: list[str] = []
        self.status: dict[str, dict[str, object]] = {}

    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        cid = str(payload["client_order_id"])
        if cid.endswith("2"):
            raise BrokerError("Leg reject")
        order_id = f"O{len(self.orders) + 1}"
        self.orders[cid] = {"order_id": order_id}
        self.status[order_id] = {"filled_qty": payload["qty"]}
        return self.orders[cid]

    def get_order_by_client_order_id(self, cid: str) -> dict[str, object] | None:
        return self.orders.get(cid)

    def cancel_order(self, order_id: str) -> None:
        self.cancelled.append(order_id)
        for cid, payload in list(self.orders.items()):
            if payload.get("order_id") == order_id:
                self.orders.pop(cid)

    def get_order_status(self, order_id: str) -> dict[str, object]:
        return self.status.get(order_id, {"filled_qty": 0})


def test_atomic_multi_leg_rejection_triggers_cancel_all() -> None:
    risk = RiskConfig(
        max_daily_trades=10, max_order_notional=2_500_000, allow_short=True
    )
    mdm = StaticMDM()
    mdm.set_quote("NIFTY", 100.0, 101.0)
    broker = RejectingBroker()
    executor = OrderExecutor(broker, risk, mdm)
    legs = [
        {
            "symbol": "NIFTY",
            "side": "BUY",
            "qty": 25,
            "price": 100.0,
            "client_order_id": "LEG1",
        },
        {
            "symbol": "NIFTY",
            "side": "SELL",
            "qty": 25,
            "price": 101.0,
            "client_order_id": "LEG2",
        },
    ]
    with pytest.raises(OrderPlacementError):
        executor.place_multi_leg_market_order(legs)
    assert broker.cancelled == ["O1"]
