from datetime import datetime
import logging

from src.execution.order_executor import OrderExecutor
from src.execution.order_state import OrderLeg, LegType, OrderSide, OrderState, TradeFSM


class DummyExe(OrderExecutor):
    def __init__(self):
        super().__init__(kite=None)
        self.logger = logging.getLogger("test")
        self.calls = []
        def place(req):
            self.calls.append(req["tradingsymbol"])
            return {"ok": True, "order_id": f"OID{len(self.calls)}"}
        self._place_with_cb = place  # type: ignore


def _leg(idx: int) -> OrderLeg:
    return OrderLeg(
        trade_id="t",
        leg_id=f"L{idx}",
        leg_type=LegType.ENTRY,
        side=OrderSide.BUY,
        symbol="SYM",
        qty=1,
        limit_price=None,
        state=OrderState.NEW,
        idempotency_key=f"k{idx}",
        created_at=datetime.utcnow(),
        expires_at=None,
        reason=None,
    )


def test_single_inflight(monkeypatch) -> None:
    exe = DummyExe()
    fsm = TradeFSM(trade_id="t", legs={})
    exe._fsms["t"] = fsm
    leg1, leg2 = _leg(1), _leg(2)
    fsm.legs["L1"] = leg1
    fsm.legs["L2"] = leg2
    exe._enqueue_router(leg1, {"bid":100, "ask":100.4}, None)
    exe._enqueue_router(leg2, {"bid":100, "ask":100.4}, None)
    exe.step_queue()
    assert exe.calls == ["SYM"]
    # complete first leg
    exe._queues["SYM"].popleft()
    exe._inflight_symbols["SYM"] = 0
    exe.step_queue()
    assert exe.calls == ["SYM", "SYM"]
