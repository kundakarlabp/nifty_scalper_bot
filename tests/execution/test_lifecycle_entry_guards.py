from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from nifty_scalper_bot.execution import native_entry_gate
from nifty_scalper_bot.execution.ownership import BoundBracketManager


class _Result:
    def __init__(self, *, accepted: bool, order_id: str | None, reason: str, details: dict[str, Any], broker_attempted: bool) -> None:
        self.accepted = accepted
        self.order_id = order_id
        self.reason = reason
        self.details = details
        self.broker_attempted = broker_attempted


class _BaseModule:
    TradePlanSubmitResult = _Result
    ManagedOrderResult = _Result


class _Manager:
    def __init__(self, provider: Any) -> None:
        self._unresolved_exit_provider = provider
        self._last_order_decision: dict[str, Any] = {}
        self.skip_reasons: list[str] = []
        self._logger = SimpleNamespace(critical=lambda *_args, **_kwargs: None)

    def set_last_skip_reason(self, reason: str) -> None:
        self.skip_reasons.append(reason)

    def is_live_mode(self) -> bool:
        return True


def _base_place_order(_manager: Any, *, intent: str | None = None, tag: str | None = None) -> None:
    return None


def test_native_entry_gate_blocks_generic_provider_reason() -> None:
    provider = SimpleNamespace(current_entry_blocker=lambda: "pnl_reconciliation_mismatch")
    manager = _Manager(provider)

    result = native_entry_gate.block_result(
        manager,
        _BaseModule,
        _base_place_order,
        "submit_trade_plan_result",
        (object(),),
        {},
    )

    assert result.accepted is False
    assert result.reason == "pnl_reconciliation_mismatch"
    assert result.broker_attempted is False
    assert result.details["provider_blocker"] is True
    assert manager._last_order_decision["block_reason"] == "pnl_reconciliation_mismatch"
    assert manager.skip_reasons == ["pnl_reconciliation_mismatch"]


def test_bound_bracket_manager_surfaces_position_manager_reason() -> None:
    position_manager = SimpleNamespace(
        current_entry_protection_blocker=lambda: None,
        current_pnl_reconciliation_blocker=lambda: "pnl_reconciliation_mismatch",
        unresolved_terminal_summary=lambda: {"count": 0},
    )
    order_manager = SimpleNamespace(_position_manager=position_manager)
    manager = BoundBracketManager.__new__(BoundBracketManager)
    manager.order_manager = order_manager
    manager.has_unresolved_exit = lambda: False

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "pnl_reconciliation_mismatch"
    assert blocker["block_source"] == "current_pnl_reconciliation_blocker"
    assert blocker["broker_attempted"] is False
