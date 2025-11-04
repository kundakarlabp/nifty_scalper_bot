"""Tests for :mod:`nifty_scalper_bot.execution.strategy_orchestrator`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pytest

from nifty_scalper_bot.execution.strategy_orchestrator import (
    StrategyOrchestrator,
    StrategyOrchestratorConfig,
)


@dataclass(slots=True)
class DummyOrderManager:
    """Minimal order manager mock capturing bracket calls."""

    orders: list[dict[str, float]]
    closed: bool = False

    def execute_bracket_trade(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        tag: str | None = None,
    ) -> tuple[str, str, str]:
        self.orders.append(
            {
                "symbol": symbol,
                "side": side,
                "quantity": float(quantity),
                "entry_price": entry_price,
                "stop_loss": stop_loss,
                "take_profit": take_profit,
            }
        )
        return ("entry", "stop", "target")

    def close_all_positions(self, *, reason: str) -> None:
        self.closed = True


@dataclass(slots=True)
class DummyBroker:
    """Broker stub returning a predefined positions list."""

    positions: Sequence[Mapping[str, object]]

    def get_positions(self) -> Sequence[Mapping[str, object]]:
        return self.positions


def test_execute_entry_converts_spot_levels() -> None:
    """Spot based SL/TP should be translated using delta."""

    manager = DummyOrderManager(orders=[])
    broker = DummyBroker(positions=[])
    config = StrategyOrchestratorConfig(
        daily_loss_limit=5.0,
        daily_profit_target=5.0,
        starting_equity=100_000.0,
    )
    orchestrator = StrategyOrchestrator(
        order_manager=manager,
        broker=broker,
        config=config,
    )

    orchestrator._execute_entry(
        symbol="NIFTY24FEB18000CE",
        side="BUY",
        quantity=50,
        entry_price=120.0,
        spot_price=22_450.0,
        sl_spot=22_400.0,
        delta=0.5,
    )

    assert len(manager.orders) == 1
    order = manager.orders[0]
    option_move = abs(22_450.0 - 22_400.0) * 0.5
    assert pytest.approx(order["stop_loss"], rel=1e-6) == 120.0 - option_move
    assert pytest.approx(order["take_profit"], rel=1e-6) == 120.0 + option_move * 2


def test_run_strategy_cycle_blocks_on_loss_limit() -> None:
    """Daily loss limit should close positions and skip processing."""

    manager = DummyOrderManager(orders=[])
    broker = DummyBroker(positions=[{"realized": -3_000.0}])
    config = StrategyOrchestratorConfig(
        daily_loss_limit=1.0,
        daily_profit_target=10.0,
        starting_equity=100_000.0,
    )
    invoked: list[str] = []

    def handler() -> None:
        invoked.append("called")

    orchestrator = StrategyOrchestrator(
        order_manager=manager,
        broker=broker,
        config=config,
        signal_handler=handler,
    )

    orchestrator.run_strategy_cycle()

    assert manager.closed is True
    assert not invoked


def test_run_strategy_cycle_blocks_on_profit_target() -> None:
    """Daily profit target should trigger protective closure."""

    manager = DummyOrderManager(orders=[])
    broker = DummyBroker(positions=[{"realized": 5_000.0}])
    config = StrategyOrchestratorConfig(
        daily_loss_limit=5.0,
        daily_profit_target=3.0,
        starting_equity=100_000.0,
    )
    invoked: list[str] = []

    orchestrator = StrategyOrchestrator(
        order_manager=manager,
        broker=broker,
        config=config,
        signal_handler=invoked.append,
    )

    orchestrator.run_strategy_cycle()

    assert manager.closed is True
    assert not invoked


def test_run_strategy_cycle_processes_when_within_limits() -> None:
    """Signal handler should run when P&L within configured limits."""

    manager = DummyOrderManager(orders=[])
    broker = DummyBroker(positions=[{"realized": 500.0}])
    config = StrategyOrchestratorConfig(
        daily_loss_limit=5.0,
        daily_profit_target=5.0,
        starting_equity=100_000.0,
    )
    invoked: list[str] = []

    def handler() -> None:
        invoked.append("run")

    orchestrator = StrategyOrchestrator(
        order_manager=manager,
        broker=broker,
        config=config,
        signal_handler=handler,
    )

    orchestrator.run_strategy_cycle()

    assert manager.closed is False
    assert invoked == ["run"]
