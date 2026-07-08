from __future__ import annotations

from types import SimpleNamespace
from threading import RLock

from nifty_scalper_bot.execution import bracket_core as legacy
from nifty_scalper_bot.execution.live_exit_reconciliation_patch import _patched_reconcile_exit_state


class _FakeManager:
    def __init__(self) -> None:
        self._lock = RLock()
        self._exit_reconcile_interval_seconds = 0.0
        self._exit_unresolved_escalation_seconds = 9999.0
        self._exit_retry_enabled = True
        self._exit_max_retry_attempts = 2
        self.closed = False
        self.order_manager = SimpleNamespace(wait_for_fill=lambda *_a, **_k: False)

    def _strict_ledger_release_required(self) -> bool:
        return True

    def _get_broker_order_status(self, _order_id: str):
        return {"status": "OPEN PENDING", "average_price": 132.50}

    def _extract_status_price(self, status):
        return float(status.get("average_price") or 0.0) if status else None

    def _position_flat_for_symbol(self, _symbol: str) -> bool:
        return True

    def _close_bracket(self, *_a, **_k) -> None:
        self.closed = True

    def _retry_delay_for_attempt(self, _attempt: int) -> float:
        return 0.1

    def _escalate_exit_locked(self, bracket, reason: str) -> None:
        bracket.exit_state = legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
        bracket.last_exit_error = reason

    def _log_exit_pending_summary_locked(self, *_a, **_k) -> None:
        return None


def _bracket() -> SimpleNamespace:
    return SimpleNamespace(
        bracket_id="BR1",
        entry_order_id="ENTRY1",
        symbol="NFO:NIFTY2671424250CE",
        exit_state=legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value,
        entry_status=legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value,
        exit_order_id="EXIT1",
        pending_exit_order_id="EXIT1",
        exit_pending=True,
        exit_in_progress=False,
        exit_attempt_count=1,
        last_exit_attempt_at=0.0,
        exit_triggered_at=0.0,
        next_exit_attempt_at=None,
        last_exit_error=None,
        remaining_quantity=65,
        quantity=65,
        position_flat_confirmed=False,
        active=True,
        updated_at=0.0,
    )


def test_live_exit_does_not_close_on_flat_when_exit_order_still_open_pending() -> None:
    manager = _FakeManager()
    bracket = _bracket()

    closed = _patched_reconcile_exit_state(manager, bracket, requested_by="post_submit")

    assert closed is False
    assert manager.closed is False
    assert bracket.exit_state == legacy.BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    assert bracket.exit_pending is True
    assert bracket.position_flat_confirmed is False


def test_live_exit_defers_flat_without_exit_identity_and_keeps_unresolved() -> None:
    manager = _FakeManager()
    ledger_blocks = {}

    def _block_ledger_release(bracket, *, reason: str, payload: dict) -> None:
        ledger_blocks[bracket.bracket_id] = {"reason": reason, "payload": payload}

    manager._block_ledger_release = _block_ledger_release
    bracket = _bracket()
    bracket.exit_order_id = None
    bracket.pending_exit_order_id = None

    closed = _patched_reconcile_exit_state(manager, bracket, requested_by="post_submit")

    assert closed is False
    assert manager.closed is False
    assert ledger_blocks["BR1"]["reason"] == "exit_flat_without_order_identity"
    assert bracket.exit_state == legacy.BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert bracket.position_flat_confirmed is True
