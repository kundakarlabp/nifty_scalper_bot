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
