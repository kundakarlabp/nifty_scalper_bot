from datetime import datetime

from src.execution.order_executor import OrderExecutor
from src.execution.order_state import (
    LegType,
    OrderLeg,
    OrderSide,
    OrderState,
)


def make_leg(key: str) -> OrderLeg:
    return OrderLeg(
        trade_id="t",
        leg_id=f"l-{key}",
        leg_type=LegType.ENTRY,
        side=OrderSide.BUY,
        symbol="SYM",
        qty=1,
        limit_price=None,
        state=OrderState.NEW,
        idempotency_key=key,
        created_at=datetime.utcnow(),
        expires_at=None,
        reason=None,
    )


def test_idempotent_enqueue() -> None:
    exe = OrderExecutor(kite=None)
    leg1 = make_leg("k1")
    leg2 = make_leg("k1")
    exe.enqueue_leg(leg1)
    exe.enqueue_leg(leg2)
    exe.step_queue(datetime.utcnow())
    assert leg1.state is OrderState.PENDING
    assert leg2.state is OrderState.NEW
