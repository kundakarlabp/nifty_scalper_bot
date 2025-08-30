from datetime import datetime, timedelta

from src.execution.order_state import LegType, OrderLeg, OrderSide, OrderState


def _make_leg(**kwargs) -> OrderLeg:
    now = datetime.utcnow()
    defaults = dict(
        trade_id="t",
        leg_id="l",
        leg_type=LegType.ENTRY,
        side=OrderSide.BUY,
        symbol="SYM",
        qty=10,
        limit_price=None,
        state=OrderState.NEW,
        idempotency_key="k",
        created_at=now,
        expires_at=now + timedelta(seconds=1),
        reason=None,
    )
    defaults.update(kwargs)
    return OrderLeg(**defaults)


def test_new_to_pending_on_ack() -> None:
    leg = _make_leg()
    leg.mark_acked("b1")
    assert leg.state is OrderState.PENDING
    assert leg.broker_order_id == "b1"


def test_partial_then_fill() -> None:
    leg = _make_leg()
    leg.mark_acked("b1")
    leg.on_partial(5, 101.0)
    assert leg.state is OrderState.PARTIAL
    assert leg.filled_qty == 5 and leg.avg_price == 101.0
    leg.on_fill(102.0)
    assert leg.state is OrderState.FILLED
    assert leg.filled_qty == leg.qty and leg.avg_price == 102.0


def test_rejected() -> None:
    leg = _make_leg()
    leg.mark_acked("b2")
    leg.on_reject("oops")
    assert leg.state is OrderState.REJECTED
    assert leg.reason == "oops"


def test_cancel_on_timeout() -> None:
    leg = _make_leg()
    leg.mark_acked("b3")
    now = leg.created_at + timedelta(seconds=2)
    assert leg.expired(now)
    leg.on_cancel("timeout")
    assert leg.state is OrderState.CANCELLED
    assert leg.reason == "timeout"
