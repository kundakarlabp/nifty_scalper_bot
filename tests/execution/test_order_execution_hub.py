"""Tests for the order execution hub integration pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import pytest

from nifty_scalper_bot.execution.execution_router import ExecutionResult
from nifty_scalper_bot.execution.order_execution_hub import OrderExecutionHub
from nifty_scalper_bot.execution.order_queue import OrderQueue, OrderRequest
from nifty_scalper_bot.execution.preflight_validator import ValidationResult

if TYPE_CHECKING:
    from nifty_scalper_bot.execution.execution_router import ExecutionRouter
    from nifty_scalper_bot.execution.lifecycle_manager import LifecycleManager
    from nifty_scalper_bot.execution.post_fill_monitor import PostFillMonitor
    from nifty_scalper_bot.execution.preflight_validator import PreFlightValidator
    from nifty_scalper_bot.execution.state_tracker import StateTracker


class DummyStateTracker:
    def __init__(self) -> None:
        self.orders: list[dict[str, object]] = []
        self._positions: dict[str, dict[str, object]] = {}

    def add_order(self, payload: dict[str, object]) -> str:
        self.orders.append(payload)
        return str(payload.get("order_id", len(self.orders)))

    def update_position(self, symbol: str, updates: dict[str, object]) -> None:
        if updates.get("delete"):
            self._positions.pop(symbol, None)
            return
        record = dict(self._positions.get(symbol, {"symbol": symbol}))
        record.update({k: v for k, v in updates.items() if k != "delete"})
        self._positions[symbol] = record

    def get_position_state(self, symbol: str) -> dict[str, object] | None:
        return self._positions.get(symbol)

    def get_open_positions(self) -> list[dict[str, object]]:
        return list(self._positions.values())


@dataclass
class DummyPreflightValidator:
    result: ValidationResult
    risk_manager: object | None = None
    regime_manager: object | None = None

    def __post_init__(self) -> None:
        self._risk_manager = self.risk_manager
        self._regime_manager = self.regime_manager

    def validate(
        self, symbol: str, *, context: dict[str, object] | None = None
    ) -> ValidationResult:
        return self.result


class DummyLifecycleManager:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.fills: list[dict[str, object]] = []
        self.exits: list[tuple[str, str]] = []

    async def start(self) -> None:
        self.started = True

    async def shutdown(self) -> None:
        self.stopped = True

    def on_fill(self, **kwargs: object) -> None:
        self.fills.append(dict(kwargs))

    def exit_at_market(self, symbol: str, reason: str) -> None:
        self.exits.append((symbol, reason))


class DummyExecutionRouter:
    def __init__(self, result: ExecutionResult) -> None:
        self.result = result
        self.calls: list[OrderRequest] = []

    def execute(self, request: OrderRequest) -> ExecutionResult:
        self.calls.append(request)
        return self.result

    def get_stats(self) -> dict[str, float]:
        return {"live_orders": float(len(self.calls))}


class SequencedExecutionRouter:
    """Return execution results in the provided order."""

    def __init__(self, results: list[ExecutionResult]) -> None:
        self._results = list(results)
        self.calls: list[OrderRequest] = []

    def execute(self, request: OrderRequest) -> ExecutionResult:
        self.calls.append(request)
        if not self._results:
            raise AssertionError("No execution results remaining")
        return self._results.pop(0)

    def get_stats(self) -> dict[str, float]:
        return {"live_orders": float(len(self.calls))}


class DummyPostFillMonitor:
    def __init__(self) -> None:
        self.started = False
        self.stopped = False
        self.reconciliations = 0

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def reconcile(self) -> None:
        self.reconciliations += 1

    def get_stats(self) -> dict[str, int]:
        return {"total_mismatches": 0, "interval_sec": 30}


@pytest.mark.asyncio
async def test_order_execution_hub_processes_request() -> None:
    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    lifecycle = DummyLifecycleManager()
    validator = DummyPreflightValidator(
        ValidationResult(allowed=True, reasons=[], blocking_level="NONE")
    )
    router = DummyExecutionRouter(
        ExecutionResult(
            status="FILLED",
            order_id="abc",
            fill_quantity=25,
            fill_price=100.5,
        )
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )
    await hub.start()
    request = OrderRequest(
        symbol="NIFTY25OCT25000CE",
        side="BUY",
        quantity=25,
        intent="ENTRY",
        metadata={"atr": 12.5, "regime": "TREND"},
    )
    hub.submit_order_request(request)
    for _ in range(20):
        if state_tracker.orders:
            break
        await asyncio.sleep(0.05)
    await hub.shutdown()
    assert router.calls, "Router should be invoked"
    assert state_tracker.orders, "Orders should be recorded"
    assert lifecycle.fills, "Lifecycle should receive on_fill"
    assert monitor.started and monitor.stopped, "Monitor should start and stop"


@pytest.mark.asyncio
async def test_order_execution_hub_preflight_rejection() -> None:
    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    lifecycle = DummyLifecycleManager()
    validator = DummyPreflightValidator(
        ValidationResult(
            allowed=False,
            reasons=[{"gate": "market_hours", "detail": "closed"}],
            blocking_level="HARD",
        )
    )
    router = DummyExecutionRouter(
        ExecutionResult(
            status="FAILED",
            order_id=None,
            fill_quantity=0,
            fill_price=None,
        )
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )
    await hub.start()
    request = OrderRequest(
        symbol="NIFTY",
        side="BUY",
        quantity=10,
        intent="ENTRY",
    )
    hub.submit_order_request(request)
    await asyncio.sleep(0.1)
    await hub.shutdown()
    assert not router.calls, "Router should not be invoked on rejection"
    assert not state_tracker.orders, "Rejected orders should not be recorded"


def test_order_execution_hub_emergency_stop() -> None:
    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    state_tracker.update_position(
        "NIFTY25OCT25000CE",
        {"quantity": 25, "symbol": "NIFTY25OCT25000CE"},
    )
    lifecycle = DummyLifecycleManager()
    validator = DummyPreflightValidator(
        ValidationResult(allowed=True, reasons=[], blocking_level="NONE")
    )
    router = DummyExecutionRouter(
        ExecutionResult(status="FILLED", order_id="1", fill_quantity=25, fill_price=0.0)
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )
    result = hub.emergency_stop()
    assert result["queue_paused"] is True
    assert lifecycle.exits and lifecycle.exits[0][0] == "NIFTY25OCT25000CE"


@pytest.mark.asyncio
async def test_order_execution_hub_exit_preserves_entry_state() -> None:
    symbol = "NIFTY25OCT25000CE"
    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    lifecycle = DummyLifecycleManager()
    validator = DummyPreflightValidator(
        ValidationResult(allowed=True, reasons=[], blocking_level="NONE")
    )
    router = SequencedExecutionRouter(
        [
            ExecutionResult(
                status="FILLED",
                order_id="entry-1",
                fill_quantity=10,
                fill_price=100.0,
            ),
            ExecutionResult(
                status="FILLED",
                order_id="exit-partial",
                fill_quantity=5,
                fill_price=101.0,
            ),
            ExecutionResult(
                status="FILLED",
                order_id="exit-final",
                fill_quantity=5,
                fill_price=102.0,
            ),
        ]
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )
    await hub.start()

    entry_request = OrderRequest(
        symbol=symbol,
        side="BUY",
        quantity=10,
        intent="ENTRY",
        metadata={"atr": 12.5, "regime": "TREND"},
    )
    hub.submit_order_request(entry_request)
    for _ in range(40):
        if state_tracker.get_position_state(symbol):
            break
        await asyncio.sleep(0.05)
    entry_state = state_tracker.get_position_state(symbol)
    assert entry_state is not None
    entry_time = entry_state.get("entry_time")
    entry_price = entry_state.get("entry_price")

    partial_exit = OrderRequest(
        symbol=symbol,
        side="SELL",
        quantity=5,
        intent="EXIT_TP1",
    )
    hub.submit_order_request(partial_exit)
    for _ in range(40):
        position = state_tracker.get_position_state(symbol)
        if position and position.get("quantity") == 5:
            break
        await asyncio.sleep(0.05)
    reduced_state = state_tracker.get_position_state(symbol)
    assert reduced_state is not None
    assert reduced_state.get("entry_time") == entry_time
    assert reduced_state.get("entry_price") == entry_price

    final_exit = OrderRequest(
        symbol=symbol,
        side="SELL",
        quantity=5,
        intent="EXIT_SL",
    )
    hub.submit_order_request(final_exit)
    for _ in range(40):
        if state_tracker.get_position_state(symbol) is None:
            break
        await asyncio.sleep(0.05)

    await hub.shutdown()

    assert state_tracker.get_position_state(symbol) is None
    assert len(router.calls) == 3


def test_order_execution_hub_detects_circuit_breaker() -> None:
    class CircuitRiskManager:
        def __init__(self) -> None:
            self.calls = 0

        def is_circuit_breaker_tripped(self) -> tuple[bool, str]:
            self.calls += 1
            return True, "daily_loss_limit"

    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    lifecycle = DummyLifecycleManager()
    risk_manager = CircuitRiskManager()
    validator = DummyPreflightValidator(
        ValidationResult(allowed=True, reasons=[], blocking_level="NONE"),
        risk_manager=risk_manager,
    )
    router = DummyExecutionRouter(
        ExecutionResult(
            status="FAILED", order_id=None, fill_quantity=0, fill_price=None
        )
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )

    assert hub._should_halt_processing() is True
    assert risk_manager.calls == 1


def test_order_execution_hub_resolves_regime_from_manager() -> None:
    class RegimeManager:
        def __init__(self) -> None:
            self.calls = 0

        def get_current_regime(self) -> str:
            self.calls += 1
            return "TRENDING"

    queue = OrderQueue()
    state_tracker = DummyStateTracker()
    lifecycle = DummyLifecycleManager()
    regime_manager = RegimeManager()
    validator = DummyPreflightValidator(
        ValidationResult(allowed=True, reasons=[], blocking_level="NONE"),
        regime_manager=regime_manager,
    )
    router = DummyExecutionRouter(
        ExecutionResult(
            status="FILLED", order_id="1", fill_quantity=1, fill_price=100.0
        )
    )
    monitor = DummyPostFillMonitor()
    hub = OrderExecutionHub(
        state_tracker=cast("StateTracker", state_tracker),
        preflight_validator=cast("PreFlightValidator", validator),
        lifecycle_manager=cast("LifecycleManager", lifecycle),
        order_queue=queue,
        execution_router=cast("ExecutionRouter", router),
        post_fill_monitor=cast("PostFillMonitor", monitor),
        data_hub=None,
    )

    regime = hub._get_current_regime("NIFTY25FEB24000CE", {})
    assert regime == "TRENDING"
    assert regime_manager.calls == 1
