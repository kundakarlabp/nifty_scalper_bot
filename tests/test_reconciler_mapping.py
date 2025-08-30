import logging
from datetime import datetime
from src.execution.order_executor import OrderExecutor, OrderReconciler
from src.execution.order_state import (
    LegType,
    OrderLeg,
    OrderSide,
    OrderState,
    TradeFSM,
)


class FakeKite:
    def __init__(self, orders):
        self._orders = orders

    def orders(self):  # pragma: no cover - simple mock
        return self._orders


def make_leg(trade: str, leg_id: str, key: str, broker_id: str | None = None) -> OrderLeg:
    leg = OrderLeg(
        trade_id=trade,
        leg_id=leg_id,
        leg_type=LegType.ENTRY,
        side=OrderSide.BUY,
        symbol="SYM",
        qty=10,
        limit_price=None,
        state=OrderState.NEW,
        idempotency_key=key,
        created_at=datetime.utcnow(),
        expires_at=None,
        reason=None,
    )
    if broker_id:
        leg.mark_acked(broker_id)
    return leg


def test_reconciler_updates_by_id_and_tag() -> None:
    exe = OrderExecutor(kite=None)
    leg1 = make_leg("t1", "l1", "k1", broker_id="oid1")
    leg2 = make_leg("t2", "l2", "k2")
    fsm1 = TradeFSM(trade_id="t1", legs={"l1": leg1})
    fsm2 = TradeFSM(trade_id="t2", legs={"l2": leg2})
    exe._fsms = {"t1": fsm1, "t2": fsm2}
    exe._idemp_map[leg2.idempotency_key] = leg2.leg_id

    kite = FakeKite(
        [
            {"order_id": "oid1", "status": "COMPLETE", "filled_quantity": 10, "average_price": 100.0},
            {"tag": "k2", "status": "REJECTED", "status_message": "reject"},
        ]
    )
    rec = OrderReconciler(kite, exe, logging.getLogger("test"))
    updated = rec.step(datetime.utcnow())
    assert updated == 2
    assert leg1.state is OrderState.FILLED
    assert leg2.state is OrderState.REJECTED
    assert fsm1.status == "CLOSED"
    assert fsm2.status == "CLOSED"
