from datetime import datetime
import logging

from src.execution.order_executor import OrderExecutor
from src.execution.order_state import OrderLeg, LegType, OrderSide, OrderState, TradeFSM


class DummyExe(OrderExecutor):
    def __init__(self):
        super().__init__(kite=None)
        self.logger = logging.getLogger("test")
        
        def place(req):
            return {"ok": True, "order_id": "OID1"}
        self._place_with_cb = place  # type: ignore
        self._modify_calls = []
        def modify(order_id, **kw):
            self._modify_calls.append(kw.get("price"))
            return {"ok": True}
        self._modify_with_cb = modify  # type: ignore
        self._cancel_calls = []
        def cancel(order_id, **kw):
            self._cancel_calls.append(order_id)
            return {"ok": True}
        self._cancel_with_cb = cancel  # type: ignore


def _make_leg() -> OrderLeg:
    return OrderLeg(
        trade_id="t",
        leg_id="L1",
        leg_type=LegType.ENTRY,
        side=OrderSide.BUY,
        symbol="SYM",
        qty=1,
        limit_price=None,
        state=OrderState.NEW,
        idempotency_key="k1",
        created_at=datetime.utcnow(),
        expires_at=None,
        reason=None,
    )


def test_ladder_and_timeouts(monkeypatch) -> None:
    exe = DummyExe()
    leg = _make_leg()
    fsm = TradeFSM(trade_id="t", legs={"L1": leg})
    exe._fsms["t"] = fsm

    assert exe._enqueue_router(leg, {"bid": 100.0, "ask": 100.4}, None)
    exe.step_queue()
    qi = exe._queues["SYM"][0]
    assert qi.placed_order_id == "OID1"
    assert exe._modify_calls == []

    # ack timeout -> modify to step1 price 100.35
    qi.placed_at_ms = exe._now_ms() - exe.router_ack_timeout_ms - 1
    exe.on_order_timeout_check()
    assert exe._modify_calls and exe._modify_calls[0] == 100.35

    # fill timeout -> cancel
    qi.acked_at_ms = exe._now_ms() - exe.router_fill_timeout_ms - 1
    exe.on_order_timeout_check()
    assert exe._cancel_calls == ["OID1"]
    assert not exe._queues["SYM"]
