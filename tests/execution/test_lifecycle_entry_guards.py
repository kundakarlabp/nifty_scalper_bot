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


def _bound_manager(position_manager: Any) -> BoundBracketManager:
    order_manager = SimpleNamespace(_position_manager=position_manager)
    manager = BoundBracketManager.__new__(BoundBracketManager)
    manager.order_manager = order_manager
    manager.has_unresolved_exit = lambda: False
    return manager


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


def test_native_entry_gate_fails_closed_when_provider_raises() -> None:
    def _broken() -> str:
        raise RuntimeError("position manager unavailable")

    manager = _Manager(SimpleNamespace(current_entry_blocker=_broken))

    result = native_entry_gate.block_result(
        manager,
        _BaseModule,
        _base_place_order,
        "submit_trade_plan_result",
        (object(),),
        {},
    )

    assert result.accepted is False
    assert result.reason == "entry_blocker_provider_error"
    assert "RuntimeError" in result.details["provider_error"]
    assert result.broker_attempted is False


def test_protective_order_is_not_stopped_by_entry_blocker() -> None:
    provider = SimpleNamespace(current_entry_blocker=lambda: "unresolved_terminal_order")
    manager = _Manager(provider)

    result = native_entry_gate.block_result(
        manager,
        _BaseModule,
        _base_place_order,
        "place_order",
        (),
        {"intent": "EXIT", "tag": "risk-reduction"},
    )

    assert result is native_entry_gate.NO_BLOCK
    assert manager._last_order_decision == {}


def test_bound_bracket_manager_surfaces_position_manager_reason() -> None:
    position_manager = SimpleNamespace(
        current_entry_protection_blocker=lambda: None,
        current_pnl_reconciliation_blocker=lambda: "pnl_reconciliation_mismatch",
        unresolved_terminal_summary=lambda: {"count": 0},
    )
    manager = _bound_manager(position_manager)

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "pnl_reconciliation_mismatch"
    assert blocker["block_source"] == "current_pnl_reconciliation_blocker"
    assert blocker["broker_attempted"] is False


def test_bound_bracket_manager_surfaces_unresolved_terminal_summary() -> None:
    position_manager = SimpleNamespace(
        current_entry_protection_blocker=lambda: None,
        current_pnl_reconciliation_blocker=lambda: None,
        current_position_reconciliation_blocker=lambda: None,
        current_orphan_position_blocker=lambda: None,
        current_exit_lifecycle_blocker=lambda: None,
        unresolved_terminal_summary=lambda: {"count": 2, "oldest_age_s": 17.5},
    )
    manager = _bound_manager(position_manager)

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "unresolved_terminal_order"
    assert blocker["unresolved_terminal_count"] == 2
    assert blocker["oldest_unresolved_terminal_age_s"] == 17.5


def test_bound_bracket_manager_blocks_unmanaged_broker_synced_positions() -> None:
    position_manager = SimpleNamespace(
        current_entry_protection_blocker=lambda: None,
        current_pnl_reconciliation_blocker=lambda: None,
        current_position_reconciliation_blocker=lambda: None,
        current_orphan_position_blocker=lambda: None,
        current_exit_lifecycle_blocker=lambda: None,
        unresolved_terminal_summary=lambda: {"count": 0},
        get_open_positions=lambda: [SimpleNamespace(symbol="NFO:NIFTY2662324050PE", order_id=None)],
    )
    manager = _bound_manager(position_manager)

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "broker_synced_unmanaged_position"
    assert blocker["unmanaged_position_count"] == 1


def test_bound_bracket_manager_blocks_unhealthy_reconciliation_state() -> None:
    position_manager = SimpleNamespace(
        current_entry_protection_blocker=lambda: None,
        current_pnl_reconciliation_blocker=lambda: None,
        unresolved_terminal_summary=lambda: {"count": 0},
        get_open_positions=lambda: [],
        _consecutive_reconcile_failures=2,
        _last_reconcile_error="fetch_error",
    )
    manager = _bound_manager(position_manager)

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "position_reconciliation_unhealthy"
    assert blocker["consecutive_reconcile_failures"] == 2
    assert blocker["last_reconcile_error"] == "fetch_error"


def test_bound_bracket_manager_keeps_unresolved_bracket_priority() -> None:
    position_manager = SimpleNamespace(current_pnl_reconciliation_blocker=lambda: "pnl_reconciliation_mismatch")
    order_manager = SimpleNamespace(_position_manager=position_manager)
    manager = BoundBracketManager.__new__(BoundBracketManager)
    manager.order_manager = order_manager
    manager.has_unresolved_exit = lambda: True
    manager.get_first_unresolved_exit_bracket_id = lambda: "bracket-1"

    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "unresolved_exit_position"
    assert blocker["bracket_id"] == "bracket-1"


def test_current_entry_blocker_is_read_only() -> None:
    broker = SimpleNamespace(get_positions=lambda: (_ for _ in ()).throw(AssertionError("broker called")))
    order_manager = SimpleNamespace(_position_manager=None, _broker=broker)
    manager = BoundBracketManager.__new__(BoundBracketManager)
    manager.order_manager = order_manager
    state = {"managed": True, "unresolved": True}
    manager.has_unresolved_exit = lambda: state["unresolved"]
    manager.get_first_unresolved_exit_bracket_id = lambda: "entry-1"
    manager.reconcile_symbol_flat = lambda _symbol: (_ for _ in ()).throw(AssertionError("reconcile called"))

    before = dict(state)
    blocker = manager.current_entry_blocker()

    assert blocker is not None
    assert blocker["block_reason"] == "unresolved_exit_position"
    assert state == before
