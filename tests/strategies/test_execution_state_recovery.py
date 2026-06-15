"""Execution-state stale recovery + symbol-identity tests (root-cause of
signal_state_rejected). Async so they execute under the repo conftest hook.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

from nifty_scalper_bot.execution.order_state_machine import ExecutionState, OrderStateMachine
from nifty_scalper_bot.strategies.runner import StrategyRunner


class _Logger:
    def info(self, *a, **k) -> None: ...
    def warning(self, *a, **k) -> None: ...
    def error(self, *a, **k) -> None: ...
    def debug(self, *a, **k) -> None: ...


def _runner(position_manager=None) -> StrategyRunner:
    r = StrategyRunner.__new__(StrategyRunner)
    r._logger = _Logger()
    r._execution_state_lock = threading.RLock()
    r._execution_state_by_symbol = {}
    r._order_pending_timeout_seconds = 15.0
    r._position_manager = position_manager
    return r


def _force_pending(machine: OrderStateMachine, *, order_id=None, age_s=0.0) -> None:
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING, order_id=order_id)
    if age_s:
        # backdate entered_at to simulate an aged state
        from datetime import datetime, timedelta, timezone
        machine._entered_at = datetime.now(timezone.utc) - timedelta(seconds=age_s)


async def test_stale_order_pending_without_active_order_recovers() -> None:
    # No pending order, no position, aged past timeout -> recovered to IDLE, signal accepted.
    pm = SimpleNamespace(get_pending_orders=lambda _s: [], has_position=lambda _s: False)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    _force_pending(machine, order_id=None, age_s=30.0)
    ok, reason, details = r._prepare_order_state_for_submission(sym, trace_id="t1")
    assert ok is True
    assert reason == "ok"
    assert details["before"]["state"] == "IDLE"  # reconciled before snapshot


async def test_fresh_order_pending_with_active_order_is_not_recovered() -> None:
    # A real, recent pending order must still block (no false recovery).
    pm = SimpleNamespace(get_pending_orders=lambda _s: [object()], has_position=lambda _s: False)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    _force_pending(machine, order_id="REAL-1", age_s=2.0)
    ok, reason, _ = r._prepare_order_state_for_submission(sym, trace_id="t2")
    assert ok is False
    assert reason == "signal_state_rejected"


async def test_aged_order_pending_with_active_order_is_not_recovered() -> None:
    # Aged but still backed by a real pending order -> must NOT recover.
    pm = SimpleNamespace(get_pending_orders=lambda _s: [object()], has_position=lambda _s: False)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    _force_pending(machine, order_id="REAL-2", age_s=120.0)
    ok, reason, _ = r._prepare_order_state_for_submission(sym, trace_id="t3")
    assert ok is False


async def test_real_open_position_is_not_force_reset() -> None:
    # POSITION_OPEN with a real position must never be auto-cleared.
    pm = SimpleNamespace(get_pending_orders=lambda _s: [], has_position=lambda _s: True)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING)
    machine.transition(ExecutionState.POSITION_OPEN)
    r._reconcile_stale_execution_state(machine, sym, trace_id="t4")
    assert machine.state == ExecutionState.POSITION_OPEN  # preserved


async def test_orphan_position_open_without_position_recovers() -> None:
    # POSITION_OPEN with no backing position -> recovered.
    pm = SimpleNamespace(get_pending_orders=lambda _s: [], has_position=lambda _s: False)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    machine.transition(ExecutionState.SIGNAL_RECEIVED)
    machine.transition(ExecutionState.ORDER_PENDING)
    machine.transition(ExecutionState.POSITION_OPEN)
    r._reconcile_stale_execution_state(machine, sym, trace_id="t5")
    assert machine.state == ExecutionState.IDLE


async def test_reconcile_fails_safe_when_position_manager_raises() -> None:
    # If reconciliation source raises, we must NOT clear a possibly-real state.
    def _boom(_s):
        raise RuntimeError("pm down")
    pm = SimpleNamespace(get_pending_orders=_boom, has_position=_boom)
    r = _runner(pm)
    sym = "NFO:NIFTY2661623900CE"
    machine = r._execution_state_by_symbol.setdefault(sym, OrderStateMachine())
    _force_pending(machine, order_id=None, age_s=120.0)
    r._reconcile_stale_execution_state(machine, sym, trace_id="t6")
    assert machine.state == ExecutionState.ORDER_PENDING  # not cleared on uncertainty


async def test_candidate_symbol_replacement_uses_same_state_key() -> None:
    # prepare + reset must resolve to the same normalized key.
    pm = SimpleNamespace(get_pending_orders=lambda _s: [], has_position=lambda _s: False)
    r = _runner(pm)
    sym = "nfo:nifty2661623950ce"  # lower-case to exercise normalization
    ok, _, _ = r._prepare_order_state_for_submission(sym, trace_id="t7")
    assert ok is True
    # The machine must be keyed by the normalized symbol.
    from nifty_scalper_bot.utils.symbols import canonical as _c  # noqa: F401
    keys = list(r._execution_state_by_symbol.keys())
    assert len(keys) == 1
    # reset under the same raw symbol must hit the same machine
    r._reset_execution_state(sym)
    assert r._execution_state_by_symbol[keys[0]].state == ExecutionState.IDLE
