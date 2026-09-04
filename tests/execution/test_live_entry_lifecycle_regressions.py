from __future__ import annotations

import time
from types import SimpleNamespace

from nifty_scalper_bot.execution.runtime_order_manager import RuntimeOrderManager
from tests.strategies.test_continuous_live_entry_eval import _selected_option_runner
from tests.strategies.test_entry_eval_coalescing import (
    _run_loop_in_thread,
    _stop_loop,
    _wait_until,
)


SYMBOL = "NFO:NIFTY2690823900PE"
ORDER_ID = "2095721522198405120"


class _Logger:
    def info(self, *_args: object, **_kwargs: object) -> None:
        return None


class _BracketAuthority:
    def __init__(self, *, status: str = "CANCELLED", flat: bool = True) -> None:
        self.status = status
        self.flat = flat

    def _broker_entry_order_status(self, order_id: str):
        assert order_id == ORDER_ID
        return {"status": self.status}, True

    def _position_flat_for_symbol(self, symbol: str) -> bool:
        assert symbol == SYMBOL
        return self.flat


def _blocked_manager(authority: object) -> RuntimeOrderManager:
    manager = RuntimeOrderManager.__new__(RuntimeOrderManager)
    blocker = {
        "block_reason": "entry_reconciliation_pending",
        "broker_attempted": True,
        "details": {
            "order_id": ORDER_ID,
            "symbol": SYMBOL,
            "entry_lifecycle_state": "ENTRY_RECONCILIATION_UNKNOWN",
        },
    }
    manager._entry_lifecycle_blocker = blocker
    manager._last_order_decision = blocker
    manager._bracket_manager = authority
    manager._logger = _Logger()
    return manager


def test_cancelled_flat_entry_releases_exact_lifecycle_blocker() -> None:
    """A cancelled entry proven flat must not block all future entries forever."""
    manager = _blocked_manager(_BracketAuthority(status="CANCELLED", flat=True))

    assert manager.current_entry_blocker() is None
    assert getattr(manager, "_entry_lifecycle_blocker", None) is None


def test_entry_lifecycle_blocker_stays_fail_closed_without_flat_broker_truth() -> None:
    manager = _blocked_manager(_BracketAuthority(status="CANCELLED", flat=False))

    blocker = manager.current_entry_blocker()

    assert blocker is manager._entry_lifecycle_blocker
    assert blocker["block_reason"] == "entry_reconciliation_pending"


def test_entry_lifecycle_blocker_stays_fail_closed_for_nonterminal_order() -> None:
    manager = _blocked_manager(_BracketAuthority(status="OPEN", flat=True))

    blocker = manager.current_entry_blocker()

    assert blocker is manager._entry_lifecycle_blocker
    assert blocker["block_reason"] == "entry_reconciliation_pending"


def test_entry_lifecycle_release_cannot_clear_a_newer_blocker() -> None:
    manager = _blocked_manager(SimpleNamespace())
    original = manager._entry_lifecycle_blocker
    newer = {
        "block_reason": "entry_reconciliation_pending",
        "broker_attempted": True,
        "details": {"order_id": "new-order", "symbol": SYMBOL},
    }

    class _RacingAuthority(_BracketAuthority):
        def _position_flat_for_symbol(self, symbol: str) -> bool:
            manager._entry_lifecycle_blocker = newer
            return super()._position_flat_for_symbol(symbol)

    manager._bracket_manager = _RacingAuthority(status="CANCELLED", flat=True)

    blocker = manager.current_entry_blocker()

    assert original is not newer
    assert blocker is newer
    assert manager._entry_lifecycle_blocker is newer


def test_selected_eval_updates_canonical_liveness_timestamp(monkeypatch) -> None:
    """Selected evaluation completion must update the timestamp read by watchdogs."""
    runner, _sm, _risk, _order, selected_ce = _selected_option_runner(monkeypatch)
    before = float(runner._last_selected_candidate_eval_completed_ts)
    loop, thread = _run_loop_in_thread()
    runner._main_loop = loop
    runner._runtime_loop_attached = True
    try:
        runner._on_tick_safe(
            {
                "symbol": selected_ce,
                "last_price": 112.0,
                "timestamp": time.time(),
                "source": "ws",
            }
        )
        assert _wait_until(
            lambda: int(runner._selected_candidate_eval_completed_count) >= 1,
            timeout=3.0,
        )
        assert float(runner._last_selected_candidate_eval_completed_ts) > before
        assert not hasattr(runner, "_last_selected_candidate_eval_completed_at")
    finally:
        _stop_loop(loop, thread)
