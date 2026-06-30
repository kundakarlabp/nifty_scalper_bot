"""Manual-intervention sync: when the broker reports a position flat (manual
square-off / auto-square-off), the lingering bracket must be dropped so the bot
stops re-adopting a phantom (observed: 23950CE re-adopted thousands of times)."""
from __future__ import annotations

from typing import Any

from nifty_scalper_bot.execution.bracket_manager import BracketManager


class _OM:
    def __init__(self) -> None:
        self._broker = None
    def place_order(self, **k: Any) -> str:
        return "x"
    def cancel_order(self, _o: str) -> bool:
        return True


def test_reconcile_symbol_flat_drops_bracket() -> None:
    mgr = BracketManager(order_manager=_OM())
    sym = "NFO:NIFTY26JUN23950CE"
    mgr.attach_orphan_position(symbol=sym, side="BUY", qty=65, entry_price=216.35)
    assert mgr.is_symbol_managed(sym) is True
    removed = mgr.reconcile_symbol_flat(sym)
    assert removed == 1
    assert mgr.is_symbol_managed(sym) is False


def test_reconcile_symbol_flat_noop_when_unmanaged() -> None:
    mgr = BracketManager(order_manager=_OM())
    assert mgr.reconcile_symbol_flat("NFO:NIFTY26JUN24000CE") == 0


def test_position_manager_flat_hook_fires_on_prune() -> None:
    from nifty_scalper_bot.execution.position_manager import PositionManager
    pm = PositionManager.__new__(PositionManager)
    import logging
    import threading
    pm._lock = threading.RLock()
    pm._logger = logging.getLogger("test")
    pm._positions = {}
    fired = []
    pm.set_on_symbols_flat(lambda syms: fired.extend(syms))
    # seed a local position, then sync against an EMPTY broker snapshot (closed)
    # minimal stub position object
    class _P:
        symbol = "NFO:NIFTY26JUN23950CE"
    pm._positions = {"NFO:NIFTY26JUN23950CE": _P()}
    def _save_state():
        return None
    pm.save_state = _save_state  # type: ignore
    pm.synchronize_with_broker([])  # broker reports nothing -> prune all
    assert "NFO:NIFTY26JUN23950CE" in fired
