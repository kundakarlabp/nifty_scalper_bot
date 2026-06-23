"""Escalation must FLATTEN a stuck exit with a MARKET order, not just freeze.

Regression: a LIMIT exit sat OPEN PENDING for 6+ minutes (EXIT_ESCALATED
reason=unresolved_timeout repeating, attempts=1) while the position stayed
exposed, because _escalate_exit_locked only logged/froze and never re-submitted.
"""
from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.bracket_manager import (
    BracketExitLifecycle,
    BracketManager,
)


class _Broker:
    def get_order_status(self, _oid: str) -> dict[str, Any]:
        return {"status": "OPEN PENDING", "average_price": 0.0}

    def get_positions(self) -> list[dict[str, Any]]:
        return [{"symbol": "NFO:NIFTY2660923100CE", "quantity": 65}]


class _OM:
    def __init__(self) -> None:
        self._broker = _Broker()
        self.placed: list[dict[str, Any]] = []
        self.cancelled: list[str] = []

    def place_order(self, **kwargs: Any) -> str:
        self.placed.append(kwargs)
        return "mkt-exit-1"

    def cancel_order(self, order_id: str) -> bool:
        self.cancelled.append(order_id)
        return True


def _bracket(mgr: BracketManager):
    mgr.register_virtual_bracket(
        order_id="entry-1",
        symbol="NFO:NIFTY2660923100CE",
        side="BUY",
        qty=65,
        price=157.0,
        sl=150.0,
        tp=170.0,
    )
    return next(iter(mgr._brackets.values()))


async def test_escalation_cancels_stuck_order_and_fires_market_exit():
    om = _OM()
    mgr = BracketManager(order_manager=om)
    b = _bracket(mgr)
    b.exit_order_id = "stuck-limit-1"          # the unfilled pending LIMIT
    b.pending_exit_order_id = "stuck-limit-1"
    b.remaining_quantity = 65

    with mgr._lock:
        mgr._escalate_exit_locked(b, "unresolved_timeout")

    # stuck limit cancelled
    assert "stuck-limit-1" in om.cancelled
    # a MARKET exit was placed, SELL (entry was BUY), full qty
    assert len(om.placed) == 1
    mkt = om.placed[0]
    assert mkt["order_type"] == "MARKET"
    assert mkt["side"] == "SELL"
    assert mkt["quantity"] == 65
    # new exit order id recorded
    assert b.exit_order_id == "mkt-exit-1"


async def test_escalation_market_exit_fires_only_once():
    om = _OM()
    mgr = BracketManager(order_manager=om)
    b = _bracket(mgr)
    b.exit_order_id = "stuck-limit-1"
    b.pending_exit_order_id = "stuck-limit-1"
    b.remaining_quantity = 65

    with mgr._lock:
        mgr._escalate_exit_locked(b, "unresolved_timeout")
    # reset state guard the way the reconcile loop would re-enter
    b.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    with mgr._lock:
        mgr._escalate_exit_locked(b, "unresolved_timeout")

    assert len(om.placed) == 1, "forced market exit must fire exactly once"
