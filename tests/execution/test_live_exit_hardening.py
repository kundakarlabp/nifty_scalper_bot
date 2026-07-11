from __future__ import annotations

from dataclasses import dataclass
import time
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.execution.adaptive_trailing import (
    AdaptiveTrailingController,
    TrailingSpec,
)
from nifty_scalper_bot.execution.bracket_manager import (
    BracketExitLifecycle,
    BracketManager,
)
from nifty_scalper_bot.execution.canonical_bracket_manager import (
    CanonicalBracketManager,
)
from nifty_scalper_bot.execution.hardened_adaptive_trailing import (
    HardenedAdaptiveTrailingController,
)
from nifty_scalper_bot.execution.hardened_bracket_manager import (
    HardenedBracketManager,
)


SYMBOL = "NFO:NIFTY2662324050PE"


@pytest.fixture(autouse=True)
def isolated_bracket_store(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("DATA_DIR", str(tmp_path))


class _Broker:
    def __init__(self, *, cancel_confirms: bool = True) -> None:
        self.cancel_confirms = cancel_confirms
        self.statuses: dict[str, str] = {"old-exit": "OPEN PENDING"}
        self.positions: list[dict[str, Any]] = [
            {"symbol": SYMBOL, "quantity": 65}
        ]
        self.cancel_calls: list[str] = []

    def get_order_status(self, order_id: str) -> dict[str, Any]:
        return {"status": self.statuses.get(order_id, "")}

    def get_positions(self) -> list[dict[str, Any]]:
        return list(self.positions)

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        self.cancel_calls.append(str(order_id))
        if self.cancel_confirms:
            self.statuses[str(order_id)] = "CANCELLED"
        return True


class _OrderManager:
    def __init__(self, broker: _Broker) -> None:
        self._broker = broker
        self._last_order_decision: dict[str, Any] = {}
        self.place_calls: list[dict[str, Any]] = []
        self.submit_plan_calls = 0
        self._next_id = 1

    def place_order(self, **kwargs: Any) -> str:
        self.place_calls.append(dict(kwargs))
        order_id = f"rescue-{self._next_id}"
        self._next_id += 1
        self._broker.statuses[order_id] = "OPEN"
        return order_id

    def cancel_order(self, order_id: str, *args: Any, **kwargs: Any) -> bool:
        return self._broker.cancel_order(order_id, *args, **kwargs)

    def submit_trade_plan_result(self, _plan: Any) -> str:
        self.submit_plan_calls += 1
        return "original-called"

    def submit_trade_plan(self, _plan: Any) -> str:
        self.submit_plan_calls += 1
        return "original-called"

    def place_managed_order_result(self, **_kwargs: Any) -> str:
        return "original-called"

    def place_managed_order(self, **_kwargs: Any) -> str:
        return "original-called"

    def set_last_skip_reason(self, reason: str) -> None:
        self.last_skip_reason = reason


class _DummyController:
    def __init__(self) -> None:
        self.entry_price = 0.0
        self.current_sl = 0.0
        self.highest_price = 0.0
        self.lowest_price = 0.0

    def on_tick(self, _tick: Any) -> None:
        return None


def _manager(*, cancel_confirms: bool = True) -> tuple[BracketManager, _OrderManager, _Broker]:
    broker = _Broker(cancel_confirms=cancel_confirms)
    order_manager = _OrderManager(broker)
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    manager._exit_open_order_timeout_seconds = 0.01
    manager._exit_cancel_confirm_timeout_seconds = 0.01
    manager._exit_cancel_poll_interval_seconds = 0.001
    manager.register_virtual_bracket(
        order_id="entry-1",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=150.0,
        sl=140.0,
        tp=175.0,
        activate_immediately=True,
    )
    return manager, order_manager, broker


def _make_stale_exit(manager: BracketManager) -> Any:
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.exit_pending = True
    bracket.exit_reason = "HARD_SL_BREACH"
    bracket.exit_state = BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    bracket.entry_status = bracket.exit_state
    bracket.exit_order_id = "old-exit"
    bracket.pending_exit_order_id = "old-exit"
    bracket.exit_attempt_count = 1
    bracket.exit_triggered_at = time.time() - 60.0
    bracket.last_exit_attempt_at = time.time() - 60.0
    return bracket


def test_canonical_imports_use_hardened_implementations() -> None:
    assert issubclass(BracketManager, CanonicalBracketManager)
    assert issubclass(CanonicalBracketManager, HardenedBracketManager)
    assert AdaptiveTrailingController is HardenedAdaptiveTrailingController


def test_stale_open_protective_exit_is_cancelled_then_replaced_once() -> None:
    manager, order_manager, broker = _manager()
    bracket = _make_stale_exit(manager)

    manager._process_exit_state(
        bracket,
        {"qty": 65, "reason": "HARD_SL_BREACH"},
        now=time.time(),
    )

    assert broker.cancel_calls == ["old-exit"]
    assert len(order_manager.place_calls) == 1
    rescue = order_manager.place_calls[0]
    assert rescue["order_type"] == "MARKET"
    assert rescue["check_risk"] is False
    assert str(rescue["tag"]).startswith("exit_")
    assert bracket.exit_order_id == "rescue-1"
    assert bracket.exit_state == BracketExitLifecycle.EXIT_ORDER_SUBMITTED.value
    assert bracket.exit_attempt_count == 2

    # A fresh replacement order is reconciled, not duplicated.
    manager._process_exit_state(
        bracket,
        {"qty": 65, "reason": "HARD_SL_BREACH"},
        now=time.time(),
    )
    assert len(order_manager.place_calls) == 1


def test_cancel_unconfirmed_escalates_once_and_state_remains_latched() -> None:
    manager, _order_manager, _broker = _manager(cancel_confirms=False)
    bracket = _make_stale_exit(manager)
    events: list[tuple[str, dict[str, Any]]] = []
    manager.set_notifier(lambda event, payload: events.append((event, dict(payload))))

    manager._process_exit_state(
        bracket,
        {"qty": 65, "reason": "HARD_SL_BREACH"},
        now=time.time(),
    )
    manager._process_exit_state(
        bracket,
        {"qty": 65, "reason": "HARD_SL_BREACH"},
        now=time.time() + 1.0,
    )

    assert bracket.exit_state == BracketExitLifecycle.EXIT_FAILED_ESCALATED.value
    assert bracket.exit_pending is True
    assert [event for event, _ in events].count("EXIT_ESCALATED") == 1


def test_unresolved_exit_blocks_entries_but_never_blocks_protective_exit() -> None:
    manager, order_manager, _broker = _manager()
    _make_stale_exit(manager)

    result = order_manager.submit_trade_plan_result(SimpleNamespace(symbol=SYMBOL))
    assert result.accepted is False
    assert result.reason == "unresolved_exit_position"
    assert result.broker_attempted is False
    assert order_manager.submit_plan_calls == 0

    normal = order_manager.place_order(
        symbol=SYMBOL,
        side="BUY",
        quantity=65,
        order_type="MARKET",
        tag="runner_entry",
    )
    assert normal is None
    assert len(order_manager.place_calls) == 0

    protective = order_manager.place_order(
        symbol=SYMBOL,
        side="SELL",
        quantity=65,
        order_type="MARKET",
        tag="exit_HAR_entry1",
        check_risk=False,
    )
    assert protective == "rescue-1"
    assert len(order_manager.place_calls) == 1


def test_virtual_trailing_stop_is_monotonic_and_stays_below_market_for_long() -> None:
    manager, _order_manager, _broker = _manager()
    bracket = manager.get_bracket("entry-1")
    assert bracket is not None
    bracket.sl_trigger_price = 140.0
    bracket.last_ltp = 160.0

    assert manager._virtual_modify_sl(bracket.virtual_sl_id, 150.03) is True
    assert bracket.sl_trigger_price == 150.05
    assert manager._virtual_modify_sl(bracket.virtual_sl_id, 149.0) is False
    assert manager._virtual_modify_sl(bracket.virtual_sl_id, 160.0) is False
    assert bracket.sl_trigger_price == 150.05


def test_actual_fill_resynchronizes_attached_trailing_controller() -> None:
    broker = _Broker()
    order_manager = _OrderManager(broker)
    manager = BracketManager(order_manager=order_manager)
    manager._running = False
    manager._watchdog_thread.join(timeout=1.0)
    controller = _DummyController()
    manager.attach_trailing_controller_factory(lambda _state: controller)
    manager.register_virtual_bracket(
        order_id="entry-fill",
        symbol=SYMBOL,
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        activate_immediately=False,
    )

    manager.confirm_entry_fill("entry-fill", 102.0)
    bracket = manager.get_bracket("entry-fill")
    assert bracket is not None
    assert controller.entry_price == bracket.entry_price == 102.0
    assert controller.current_sl == bracket.sl_trigger_price
    assert controller.highest_price == 102.0
    assert controller.lowest_price == 102.0


@dataclass
class _AtrSnapshot:
    value: float
    fresh: bool = True

    def is_fresh(self, max_age_sec: float) -> bool:
        assert max_age_sec == 60.0
        return self.fresh


class _AtrProvider:
    def __init__(self, snapshot: _AtrSnapshot) -> None:
        self.snapshot = snapshot

    def get_atr(self, _symbol: str, fallback: float) -> _AtrSnapshot:
        assert fallback > 0
        return self.snapshot


def test_adaptive_trailing_honours_configured_activation_and_tick_size() -> None:
    updates: list[float] = []
    controller = AdaptiveTrailingController(
        symbol=SYMBOL,
        side="LONG",
        entry=100.0,
        sl_order_id="vsl-1",
        variety="virtual",
        spec=TrailingSpec(trail_by=2.0, step=0.05, activation=1.0),
        get_ltp=lambda _symbol: 100.0,
        modify_order=lambda _order_id, price: updates.append(price) or True,
        atr_provider=_AtrProvider(_AtrSnapshot(1.0)),
        journal=SimpleNamespace(set=lambda *_args, **_kwargs: None),
        atr_multiplier=1.0,
    )
    controller.current_sl = 90.0

    controller.on_tick({"ltp": 100.5})
    assert controller.trailing_active is False
    assert updates == []

    controller.on_tick({"ltp": 101.2})
    assert controller.trailing_active is True
    assert updates
    assert round(updates[-1] / 0.05) * 0.05 == updates[-1]


def test_stale_atr_degrades_to_bounded_fallback_instead_of_disabling_trailing() -> None:
    updates: list[float] = []
    controller = AdaptiveTrailingController(
        symbol=SYMBOL,
        side="LONG",
        entry=100.0,
        sl_order_id="vsl-2",
        variety="virtual",
        spec=TrailingSpec(trail_by=20.0, step=0.05, activation=0.3),
        get_ltp=lambda _symbol: 102.0,
        modify_order=lambda _order_id, price: updates.append(price) or True,
        atr_provider=_AtrProvider(_AtrSnapshot(1.0, fresh=False)),
        journal=SimpleNamespace(set=lambda *_args, **_kwargs: None),
        atr_multiplier=2.0,
    )
    controller.current_sl = 80.0
    controller.on_tick({"ltp": 102.0})

    assert controller.trailing_active is True
    assert updates
    assert 80.0 < updates[-1] < 102.0


def test_flat_fill_latency_defers_quietly_and_rescue_skips(monkeypatch, tmp_path):
    """2026-07-10 15:00 lifecycle: broker went FLAT while the exit order status
    still read OPEN PENDING (normal 1-3s propagation lag). Within the grace
    window the reconcile must defer at INFO (not WARNING), stamp
    _flat_nonterminal_since, and the stale-order rescue must SKIP instead of
    cancel-racing an already-filled order; once the status turns COMPLETE the
    bracket closes. Exactly one exit order end-to-end."""
    import logging

    from nifty_scalper_bot.execution.bracket_manager import BracketManager

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    status = {"value": "OPEN PENDING", "flat": False}
    orders: list = []

    class _OM:
        def place_reduce_only_exit(self, intent):
            orders.append("exit")
            return "EXIT-1"

        def place_order(self, **kwargs):
            orders.append("exit")
            return "EXIT-1"

        def get_order_status(self, _oid):
            return {
                "status": status["value"],
                "average_price": 102.90 if status["value"] == "COMPLETE" else None,
            }

    om = _OM()

    class _BrokerStub:
        def get_positions(self):
            if status["flat"]:
                return []
            return [{"symbol": "NFO:NIFTYLAT24200PE", "quantity": 65}]

        def get_order_status(self, oid):
            return om.get_order_status(oid)

    om._broker = _BrokerStub()
    bm = BracketManager(order_manager=om)
    bm._running = False
    bm._exit_reconcile_interval_seconds = 0.0
    bm._filled_position_sync_grace_seconds = 0.0
    try:
        bm.register_virtual_bracket(
            order_id="lat-1", symbol="NFO:NIFTYLAT24200PE", side="BUY", qty=65,
            price=103.30, sl=97.85, tp=115.45, activate_immediately=False,
        )
        bm.confirm_entry_fill("lat-1", 103.75)
        bracket = bm.get_bracket("lat-1")
        for px in (104.5, 105.3, 106.15, 102.95):
            bm.on_tick("NFO:NIFTYLAT24200PE", px)
        assert bracket.exit_pending is True and orders == ["exit"]

        # Broker flat, order status lagging: force the authoritative flat view
        # and the strict-live reconcile path (the branch production runs).
        status["flat"] = True
        monkeypatch.setattr(
            type(bm), "_position_flat_for_symbol", lambda self, s: True
        )
        records: list = []

        class _Cap(logging.Handler):
            def emit(self, record):
                records.append(record)

        from nifty_scalper_bot.execution import bracket_core

        _prev_level = bracket_core.LOGGER.level
        bracket_core.LOGGER.setLevel(logging.INFO)
        bracket_core.LOGGER.addHandler(_Cap())
        try:
            result = bm._reconcile_exit_state(bracket, requested_by="watchdog")
        finally:
            bracket_core.LOGGER.setLevel(_prev_level)
            bracket_core.LOGGER.handlers = [
                h for h in bracket_core.LOGGER.handlers if not isinstance(h, _Cap)
            ]
        assert result is False
        assert bracket.flat_nonterminal_since_monotonic is not None
        assert bracket.flat_nonterminal_since_utc is not None
        lag_logs = [r for r in records if "EXIT_FLAT_BUT_ORDER_NOT_TERMINAL" in r.getMessage()]
        assert lag_logs and all(r.levelno == logging.INFO for r in lag_logs), [
            (r.levelname, r.getMessage()[:60]) for r in lag_logs
        ]

        # Rescue must skip while flat inside the grace window (no cancel race).
        bm._rescue_stale_exit_order(
            bracket, order_id="EXIT-1", qty=65, status="OPEN PENDING"
        )
        assert orders == ["exit"]
        assert bracket.exit_in_progress is False

        # Status propagates -> closure completes with the real fill price.
        status["value"] = "COMPLETE"
        closed = bm._reconcile_exit_state(bracket, requested_by="watchdog")
        if not closed:  # strict-live filled handoff may need one follow-up pass
            closed = bm._reconcile_exit_state(bracket, requested_by="watchdog")
        assert closed is True
        assert bracket.exit_state == "CLOSED"
        assert orders == ["exit"]
    finally:
        bm._running = False
