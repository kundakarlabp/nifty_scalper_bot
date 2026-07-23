from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.position_manager import Position, PositionManager
from nifty_scalper_bot.strategies.runner import StrategyRunner

SYMBOL = "NFO:NIFTY2660923100CE"
QTY = 65


@pytest.fixture(autouse=True)
def isolated_state(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


class _Broker:
    def __init__(
        self, *, status: str = "COMPLETE", positions: list[dict[str, Any]] | None = None
    ) -> None:
        self.status = status
        self.positions = positions if positions is not None else []

    def get_order_status(self, _order_id: str) -> dict[str, Any]:
        return {"status": self.status, "average_price": 120.0}

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)


class _OrderManager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self.calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.calls.append(kwargs)
        return "exit-1"


def _position_manager(tmp_path) -> PositionManager:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-1")
    pm.add_pending_order(
        "exit-1",
        SYMBOL,
        "SELL",
        QTY,
        120.0,
        "MARKET",
        intent="EXIT",
        bracket_id="entry-1",
    )
    return pm


def _bracket_manager() -> BracketManager:
    broker = _Broker(status="COMPLETE", positions=[])
    bm = BracketManager(order_manager=_OrderManager(broker))
    bm.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=QTY,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    bm.confirm_entry_fill("entry-1", 100.0)
    return bm


def test_exit_fill_reconciliation_flat_does_not_leave_orphan_bracket(tmp_path) -> None:
    """Terminal SELL fill makes broker, PM and bracket ownership flat immediately."""
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()

    pm.update_order_status("exit-1", "FILLED", 120.0)
    removed = bm.reconcile_symbol_flat(SYMBOL)

    assert pm.get_position(SYMBOL) is None
    assert pm.get_all_positions() == []
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert removed == 1
    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket("entry-1") is None

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    runner._adopt_orphan_positions()

    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None


def test_stale_snapshot_after_exit_fill_is_not_orphan_adopted_then_zero_clears(
    tmp_path,
) -> None:
    """A stale non-zero broker snapshot after a terminal exit cannot re-open/adopt."""
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()
    bm.order_manager._positions = pm
    adopt_calls: list[dict[str, Any]] = []
    bm.attach_orphan_position = lambda **kwargs: adopt_calls.append(kwargs) or "orphan"  # type: ignore[method-assign]

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)

    pm.synchronize_with_broker(
        [
            {
                "symbol": SYMBOL,
                "quantity": QTY,
                "average_price": 100.0,
                "last_price": 120.0,
                "product": "MIS",
            }
        ]
    )

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    runner._adopt_orphan_positions()

    assert pm.get_all_positions() == []
    assert adopt_calls == []
    assert SYMBOL in pm._recently_flat_exit_until_monotonic

    pm.synchronize_with_broker([])

    assert pm.get_all_positions() == []
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert bm.is_symbol_managed(SYMBOL) is False


def test_terminal_exit_order_not_pending_for_bracket_ghost_or_orphan(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)

    assert "exit-1" not in pm._orders
    assert pm._terminal_orders["exit-1"].lifecycle_resolved is True
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert bm.has_unresolved_exit() is False
    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.get_bracket("entry-1") is None
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None


def test_exit_reconciliation_grace_configuration(monkeypatch) -> None:
    from nifty_scalper_bot.execution import position_manager as module

    monkeypatch.delenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", raising=False)
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "bad")
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "nan")
    assert module._resolve_exit_reconciliation_grace_seconds() == 2.0
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "0.1")
    assert module._resolve_exit_reconciliation_grace_seconds() == 0.25
    monkeypatch.setenv("EXIT_RECONCILIATION_SETTLEMENT_GRACE_SECONDS", "10")
    assert module._resolve_exit_reconciliation_grace_seconds() == 5.0


def _broker_row(quantity: int = QTY) -> dict[str, Any]:
    return {
        "symbol": SYMBOL,
        "quantity": quantity,
        "average_price": 100.0,
        "last_price": 120.0,
        "product": "MIS",
    }


def test_repeated_stale_snapshots_do_not_extend_fixed_exit_deadline(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm._recently_flat_exit_grace_seconds = 1.0
    pm.update_order_status("exit-1", "FILLED", 120.0)
    first_deadline = pm._recently_flat_exit_until_monotonic[SYMBOL]

    pm.synchronize_with_broker([_broker_row()])
    pm.synchronize_with_broker([_broker_row()])

    assert pm.get_position(SYMBOL) is None
    assert pm._recently_flat_exit_until_monotonic[SYMBOL] == first_deadline
    guard = pm._recently_flat_exit_metadata[SYMBOL]
    assert guard.stale_snapshot_count == 2
    assert guard.last_stale_quantity == QTY


def test_zero_or_missing_snapshot_clears_exit_guard(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    pm.synchronize_with_broker([_broker_row(0)])
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata

    pm = _position_manager(tmp_path / "second")
    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm.synchronize_with_broker([])
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata


def test_persistent_non_zero_position_is_restored_after_exit_grace_expiry(
    tmp_path,
) -> None:
    pm = _position_manager(tmp_path)
    pm._recently_flat_exit_grace_seconds = 0.25
    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm._recently_flat_exit_until_monotonic[SYMBOL] = 0.0
    pm._recently_flat_exit_metadata[SYMBOL].grace_until_monotonic = 0.0

    pm.synchronize_with_broker([_broker_row()])

    restored = pm.get_position(SYMBOL)
    assert restored is not None
    assert restored.quantity == QTY
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic


def test_new_entry_clears_old_exit_guard(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    pm.open_position(SYMBOL, "LONG", QTY, 121.0, order_id="entry-2")

    assert SYMBOL not in pm._recently_flat_exit_until_monotonic
    assert SYMBOL not in pm._recently_flat_exit_metadata
    assert pm.get_position(SYMBOL) is not None


def test_partial_exit_does_not_create_flat_guard(tmp_path) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY * 2, 100.0, order_id="entry-1")
    pm.add_pending_order(
        "exit-partial",
        SYMBOL,
        "SELL",
        QTY,
        120.0,
        "MARKET",
        intent="REDUCE",
        bracket_id="entry-1",
    )

    pm.update_order_status("exit-partial", "FILLED", 120.0)

    assert pm.get_position(SYMBOL) is not None
    assert SYMBOL not in pm._recently_flat_exit_until_monotonic


def test_duplicate_terminal_callbacks_remain_idempotent(tmp_path) -> None:
    pm = _position_manager(tmp_path)

    pm.update_order_status("exit-1", "FILLED", 120.0)
    pm.update_order_status("exit-1", "FILLED", 120.0)

    assert pm.get_position(SYMBOL) is None
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert pm._terminal_orders["exit-1"].lifecycle_applied is True


def test_persistent_position_after_grace_gets_orphan_protection(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()
    pm._recently_flat_exit_grace_seconds = 0.25

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)
    pm.synchronize_with_broker([_broker_row()])
    assert pm.get_position(SYMBOL) is None
    assert bm.is_symbol_managed(SYMBOL) is False

    pm._recently_flat_exit_until_monotonic[SYMBOL] = 0.0
    pm._recently_flat_exit_metadata[SYMBOL].grace_until_monotonic = 0.0
    pm.synchronize_with_broker([_broker_row()])
    restored = pm.get_position(SYMBOL)
    assert restored is not None
    assert restored.quantity == QTY

    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )

    runner._adopt_orphan_positions()

    assert bm.is_symbol_managed(SYMBOL) is True
    assert bm.get_bracket(f"orphan_{SYMBOL}") is not None


def test_synchronous_exit_fill_callback_is_registered_as_exit_before_submit_returns(
    tmp_path,
) -> None:
    """Exit intent exists before broker submission, so an immediate fill is not unknown."""
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-sync")

    class SyncFillOrderManager:
        def __init__(self) -> None:
            self._positions = pm
            self._broker = _Broker(status="COMPLETE", positions=[_broker_row()])
            self.kwargs: dict[str, Any] | None = None

        def place_order(self, **kwargs: Any) -> str:
            self.kwargs = dict(kwargs)
            pm.apply_broker_order_update(
                "exit-sync-final",
                {
                    "order_id": "exit-sync-final",
                    "tag": kwargs.get("tag"),
                    "symbol": SYMBOL,
                    "side": "SELL",
                    "quantity": QTY,
                    "filled_quantity": QTY,
                    "average_price": 120.0,
                    "status": "COMPLETE",
                },
            )
            return "exit-sync-final"

    om = SyncFillOrderManager()
    bm = BracketManager(order_manager=om)
    bm.register_virtual_bracket(
        order_id="entry-sync",
        symbol=SYMBOL,
        side="BUY",
        qty=QTY,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    bm.confirm_entry_fill("entry-sync", 100.0)
    bracket = bm.get_bracket("entry-sync")
    assert bracket is not None
    bracket.last_ltp = 120.0

    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=123.0)

    assert om.kwargs is not None
    assert om.kwargs["intent"] == "EXIT"
    assert om.kwargs["bracket_id"] == "entry-sync"
    assert om.kwargs["tag"].startswith("exit_")
    assert "client_order_id" not in om.kwargs
    assert pm.get_position(SYMBOL) is None
    assert not pm.get_pending_orders(SYMBOL)
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert all(
        row.get("classification") != "BROKER_UNKNOWN_ORDER_RESOLVED"
        for row in getattr(pm, "_broker_order_ledger", {}).values()
    )
    assert bracket.exit_order_id == "exit-sync-final"
    assert bracket.pending_exit_order_id == "exit-sync-final"
    assert bracket.exit_submission_inflight is False


def _runner_for(pm: PositionManager, bm: BracketManager) -> StrategyRunner:
    runner = StrategyRunner.__new__(StrategyRunner)
    runner._position_manager = pm
    runner._bracket_manager = bm
    runner._active_orphan_guards = set()
    runner._orphan_retry_count = {}
    runner._orphan_retry_last_attempt = {}
    runner._logger = SimpleNamespace(
        debug=lambda *a, **k: None,
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    return runner


def test_orphan_scan_skips_stale_positive_position_during_exit_convergence(tmp_path) -> None:
    pm = _position_manager(tmp_path)
    bm = _bracket_manager()
    bm.order_manager._positions = pm
    adopt_calls: list[dict[str, Any]] = []
    bm.attach_orphan_position = lambda **kwargs: adopt_calls.append(kwargs) or "orphan"  # type: ignore[method-assign]

    pm.update_order_status("exit-1", "FILLED", 120.0)
    bm.reconcile_symbol_flat(SYMBOL)
    pm._positions[SYMBOL] = Position(
        symbol=SYMBOL,
        side="LONG",
        quantity=QTY,
        entry_price=100.0,
        entry_time=datetime.now(timezone.utc),
        current_price=120.0,
        order_id="stale-local",
    )

    _runner_for(pm, bm)._adopt_orphan_positions()

    assert adopt_calls == []
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None


class _FailingExitOrderManager:
    def __init__(self, pm: PositionManager, *, raises: bool = False) -> None:
        self._positions = pm
        self._broker = _Broker(status="OPEN", positions=[_broker_row()])
        self.raises = raises

    def place_order(self, **_kwargs: Any) -> str | None:
        if self.raises:
            raise RuntimeError("broker down")
        return None


def _bracket_with_pm(pm: PositionManager, om: Any) -> BracketManager:
    bm = BracketManager(order_manager=om)
    bm.register_virtual_bracket(
        order_id="entry-fail",
        symbol=SYMBOL,
        side="BUY",
        qty=QTY,
        price=100.0,
        sl=90.0,
        tp=120.0,
    )
    bm.confirm_entry_fill("entry-fail", 100.0)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None
    bracket.last_ltp = 120.0
    return bm


def test_exit_submission_rejection_removes_provisional_order(tmp_path) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-fail")
    bm = _bracket_with_pm(pm, _FailingExitOrderManager(pm))
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=125.0)

    assert not pm.get_pending_orders(SYMBOL)
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert pm.get_position(SYMBOL) is not None


def test_exit_submission_exception_removes_provisional_order(tmp_path) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-fail")
    bm = _bracket_with_pm(pm, _FailingExitOrderManager(pm, raises=True))
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=126.0)

    assert not pm.get_pending_orders(SYMBOL)
    assert pm.unresolved_terminal_summary()["count"] == 0
    assert pm.get_position(SYMBOL) is not None


class _RegistrationFailingPositionManager(PositionManager):
    def add_pending_order(self, *args: Any, **kwargs: Any) -> None:  # type: ignore[override]
        raise RuntimeError("registration unavailable")


class _RecordingExitOrderManager:
    def __init__(self, pm: Any | None) -> None:
        if pm is not None:
            self._positions = pm
        self._broker = _Broker(status="OPEN", positions=[_broker_row()])
        self.calls: list[dict[str, Any]] = []

    def place_order(self, **kwargs: Any) -> str:
        self.calls.append(dict(kwargs))
        return f"exit-final-{len(self.calls)}"


class _BindFailPositionManager(PositionManager):
    def bind_pending_order_id(self, provisional_order_id: str, final_order_id: str) -> None:  # type: ignore[override]
        raise RuntimeError("bind store unavailable")


def test_provisional_registration_failure_prevents_broker_submission(
    tmp_path, caplog
) -> None:
    pm = _RegistrationFailingPositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-regfail")
    om = _RecordingExitOrderManager(pm)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    caplog.set_level(logging.ERROR)
    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=130.0)

    assert om.calls == []
    assert pm.get_position(SYMBOL) is not None
    assert not pm.get_pending_orders(SYMBOL)
    assert bracket.exit_submission_inflight is False
    assert bracket.exit_correlation_id is None
    assert bracket.last_exit_error == "registration unavailable"
    assert "EXIT_PROVISIONAL_REGISTRATION_FAILED" in caplog.text


def test_missing_provisional_registrar_prevents_broker_submission(tmp_path, caplog) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-missing")
    om = _RecordingExitOrderManager(None)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    caplog.set_level(logging.ERROR)
    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=131.0)

    assert om.calls == []
    assert pm.get_position(SYMBOL) is not None
    assert bracket.exit_submission_inflight is False
    assert bracket.exit_correlation_id is None
    assert bracket.last_exit_error == "PositionManager.add_pending_order unavailable"
    assert "EXIT_PROVISIONAL_REGISTRATION_FAILED" in caplog.text


def test_final_id_binding_failure_stays_converging_and_blocks_orphan(
    tmp_path, caplog
) -> None:
    pm = _BindFailPositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-bindfail")
    om = _RecordingExitOrderManager(pm)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None
    adopt_calls: list[dict[str, Any]] = []
    bm.attach_orphan_position = lambda **kwargs: adopt_calls.append(kwargs) or "orphan"  # type: ignore[method-assign]

    caplog.set_level(logging.ERROR)
    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=132.0)

    assert len(om.calls) == 1
    assert bracket.exit_order_id == "exit-final-1"
    assert bracket.pending_exit_order_id == "exit-final-1"
    assert bracket.exit_correlation_id is not None
    assert bracket.exit_intent == "EXIT"
    assert bracket.last_exit_error and "exit_order_id_bind_failed" in bracket.last_exit_error
    assert bm.is_exit_converging(SYMBOL) is True
    _runner_for(pm, bm)._adopt_orphan_positions()
    assert adopt_calls == []
    assert "EXIT_ORDER_ID_BIND_FAILED" in caplog.text


def test_binding_failure_recovers_when_tagged_terminal_update_reconciles(tmp_path) -> None:
    pm = _BindFailPositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-bindrecover")
    om = _RecordingExitOrderManager(pm)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=133.0)
    correlation_id = bracket.exit_correlation_id
    assert correlation_id

    PositionManager.bind_pending_order_id(pm, correlation_id, "exit-final-1")
    pm.apply_broker_order_update(
        "exit-final-1",
        {
            "order_id": "exit-final-1",
            "tag": correlation_id,
            "symbol": SYMBOL,
            "side": "SELL",
            "quantity": QTY,
            "filled_quantity": QTY,
            "average_price": 120.0,
            "status": "COMPLETE",
        },
    )
    bm.order_manager._broker.status = "COMPLETE"
    bm.order_manager._broker.positions = []
    pm.synchronize_with_broker([])
    bm._close_bracket(bracket, close_source="test_recovery", exit_price=120.0)

    assert pm.get_position(SYMBOL) is None
    assert bm.is_symbol_managed(SYMBOL) is False
    assert bm.is_exit_converging(SYMBOL) is False
    assert bracket.exit_correlation_id is None
    assert bracket.exit_intent is None
    assert bracket.expected_exit_side is None
    assert bracket.expected_exit_qty == 0
    assert bm.get_bracket(f"orphan_{SYMBOL}") is None
    assert pm.unresolved_terminal_summary()["count"] == 0


def test_successful_exit_reconciliation_clears_convergence_metadata(tmp_path) -> None:
    pm = PositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-success-clear")
    om = _RecordingExitOrderManager(pm)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    def place_order(**kwargs: Any) -> str:
        om.calls.append(dict(kwargs))
        pm.apply_broker_order_update(
            "exit-final-1",
            {
                "order_id": "exit-final-1",
                "tag": kwargs.get("tag"),
                "symbol": SYMBOL,
                "side": "SELL",
                "quantity": QTY,
                "filled_quantity": QTY,
                "average_price": 120.0,
                "status": "COMPLETE",
            },
        )
        return "exit-final-1"

    om.place_order = place_order  # type: ignore[method-assign]
    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=time.time())
    bm.order_manager._broker.positions = []
    pm.synchronize_with_broker([])
    bm._close_bracket(bracket, close_source="test_success", exit_price=120.0)

    assert bracket.exit_submission_inflight is False
    assert bracket.exit_intent is None
    assert bracket.expected_exit_side is None
    assert bracket.expected_exit_qty == 0
    assert bracket.exit_correlation_id is None
    assert bm.is_exit_converging(SYMBOL) is False


def test_registration_failure_can_retry_without_duplicate_broker_exit(tmp_path) -> None:
    pm = _RegistrationFailingPositionManager(str(tmp_path / "positions.json"))
    pm.open_position(SYMBOL, "LONG", QTY, 100.0, order_id="entry-retry")
    om = _RecordingExitOrderManager(pm)
    bm = _bracket_with_pm(pm, om)
    bracket = bm.get_bracket("entry-fail")
    assert bracket is not None

    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=135.0)
    assert om.calls == []

    pm.add_pending_order = PositionManager.add_pending_order.__get__(pm, PositionManager)  # type: ignore[method-assign]
    bracket.next_exit_attempt_at = None
    bm._process_exit_state(bracket, {"reason": "TARGET", "qty": QTY}, now=136.0)

    assert len(om.calls) == 1
    assert bracket.exit_order_id == "exit-final-1"
