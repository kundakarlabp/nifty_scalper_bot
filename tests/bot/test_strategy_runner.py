from __future__ import annotations

import sys
import time
import types
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

mock_kite = types.ModuleType("kiteconnect")
mock_kite.KiteException = Exception
mock_kite.KiteTicker = object
sys.modules.setdefault("kiteconnect", mock_kite)

from nifty_scalper_bot.bot.components import (  # noqa: E402
    Position,
    StrategyRunner,
    StrategyRunnerConfig,
    TradeSignal,
)


class StubMarketDataManager:
    def __init__(self) -> None:
        self.prices: dict[str, float] = {}

    def set_price(self, symbol: str, price: float) -> None:
        self.prices[symbol] = price

    def get_latest_price(self, symbol: str) -> float:
        return self.prices[symbol]

    def subscribe(
        self, symbol: str
    ) -> None:  # pragma: no cover - interface placeholder
        self.prices.setdefault(symbol, 0.0)

    def unsubscribe(
        self, symbol: str
    ) -> None:  # pragma: no cover - interface placeholder
        self.prices.pop(symbol, None)

    def start(self) -> None:  # pragma: no cover - interface placeholder
        return None

    def stop(self) -> None:  # pragma: no cover - interface placeholder
        return None


class StubIndicatorEngine:
    def calculate(
        self, symbol: str, price: float
    ) -> dict[str, float]:  # pragma: no cover
        return {"price": price}


class StubStrategyManager:
    def __init__(self) -> None:
        self._signals: list[TradeSignal] = []

    def set_signals(self, signals: list[TradeSignal]) -> None:
        self._signals = list(signals)

    def generate_signals(self, symbol: str, price: float) -> list[TradeSignal]:
        return list(self._signals)


class StubRiskManager:
    def __init__(self) -> None:
        self.allowed = True
        self.calls: list[tuple[TradeSignal, float]] = []

    def allow_order(self, signal: TradeSignal, price: float) -> bool:
        self.calls.append((signal, price))
        return self.allowed


class StubOrderManager:
    def __init__(self) -> None:
        self.orders: list[tuple[str, TradeSignal, float]] = []

    def submit_order(self, signal: TradeSignal, price: float) -> str:
        order_id = f"order-{len(self.orders) + 1}"
        self.orders.append((order_id, signal, price))
        return order_id


class StubPositionManager:
    def __init__(self) -> None:
        self.positions: dict[str, Position] = {}

    def get_all_positions(self) -> list[Position]:
        return list(self.positions.values())

    def record_position(self, symbol: str, quantity: int, price: float) -> None:
        self.positions[symbol] = Position(
            symbol=symbol, quantity=quantity, average_price=price
        )

    def close_position(self, symbol: str) -> None:
        self.positions.pop(symbol, None)


def create_runner(
    config: StrategyRunnerConfig | None = None,
) -> tuple[
    StrategyRunner,
    StubMarketDataManager,
    StubStrategyManager,
    StubRiskManager,
    StubOrderManager,
    StubPositionManager,
]:
    market_data = StubMarketDataManager()
    indicator_engine = StubIndicatorEngine()
    strategy = StubStrategyManager()
    risk = StubRiskManager()
    orders = StubOrderManager()
    positions = StubPositionManager()
    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator_engine,
        signal_generator=strategy,
        risk_manager=risk,
        order_manager=orders,
        position_manager=positions,
        config=config,
    )
    return runner, market_data, strategy, risk, orders, positions


def test_should_generate_signal_respects_cooldown() -> None:
    config = StrategyRunnerConfig(signal_cooldown=5.0)
    runner, *_ = create_runner(config)
    assert runner._should_generate_signal("ABC") is True  # type: ignore[attr-defined]
    runner._last_signal_time["ABC"] = time.time()  # type: ignore[attr-defined]
    assert runner._should_generate_signal("ABC") is False  # type: ignore[attr-defined]
    runner._last_signal_time["ABC"] = time.time() - 6  # type: ignore[attr-defined]
    assert runner._should_generate_signal("ABC") is True  # type: ignore[attr-defined]


def test_execute_signal_updates_metrics_and_positions() -> None:
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(signal_cooldown=0.0)
    )
    open_positions: dict[str, Position] = {}
    runner._execute_signal(TradeSignal("ABC", "BUY", 1), 100.0, open_positions)  # type: ignore[attr-defined]
    assert "ABC" in runner._position_state  # type: ignore[attr-defined]
    assert positions.get_all_positions()
    pos = positions.get_all_positions()[0]
    open_positions = {pos.symbol: pos}
    runner._execute_signal(TradeSignal("ABC", "SELL", 1), 110.0, open_positions)  # type: ignore[attr-defined]
    metrics = runner.get_performance_metrics()
    assert metrics["total_trades"] == pytest.approx(1.0)
    assert metrics["net_profit"] == pytest.approx(10.0)
    assert len(orders.orders) == 2


def test_check_exit_conditions_triggers_stop_loss() -> None:
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(signal_cooldown=0.0)
    )
    market_data.set_price("ABC", 100.0)
    open_positions: dict[str, Position] = {}
    runner._execute_signal(
        TradeSignal("ABC", "BUY", 1, stop_loss=95.0), 100.0, open_positions
    )  # type: ignore[attr-defined]
    pos = positions.get_all_positions()[0]
    open_positions = {pos.symbol: pos}
    runner._position_state["ABC"]["stop_loss"] = 95.0  # type: ignore[attr-defined]
    triggered = runner._check_exit_conditions("ABC", 94.0, pos, open_positions)  # type: ignore[attr-defined]
    assert triggered is True
    assert positions.get_all_positions() == []
    metrics = runner.get_performance_metrics()
    assert metrics["losing_trades"] == pytest.approx(1.0)


def test_update_trailing_stop_adjusts_on_favourable_move() -> None:
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(signal_cooldown=0.0, trailing_stop_pct=0.1)
    )
    open_positions: dict[str, Position] = {}
    runner._execute_signal(TradeSignal("XYZ", "BUY", 1), 100.0, open_positions)  # type: ignore[attr-defined]
    pos = positions.get_all_positions()[0]
    runner._update_trailing_stop("XYZ", 120.0, pos)  # type: ignore[attr-defined]
    state = runner._position_state["XYZ"]  # type: ignore[attr-defined]
    assert state["trailing_stop"] == pytest.approx(108.0)
    assert state["stop_loss"] == pytest.approx(108.0)


def test_close_all_positions_before_market_close_liquidates() -> None:
    close_time = (datetime.now(ZoneInfo("UTC")) - timedelta(minutes=1)).time()
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(
            signal_cooldown=0.0,
            market_close_time=close_time,
            market_close_timezone="UTC",
            market_close_buffer_minutes=0,
        )
    )
    open_positions: dict[str, Position] = {}
    runner._execute_signal(TradeSignal("ABC", "BUY", 1), 100.0, open_positions)  # type: ignore[attr-defined]
    pos = positions.get_all_positions()[0]
    open_positions = {pos.symbol: pos}
    market_data.set_price("ABC", 101.0)
    assert runner._close_all_positions_before_market_close(open_positions) is True  # type: ignore[attr-defined]
    assert positions.get_all_positions() == []
    assert runner.get_performance_metrics()["total_trades"] == pytest.approx(1.0)


def test_on_tick_enforces_cooldown() -> None:
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(signal_cooldown=10.0)
    )
    market_data.set_price("ABC", 100.0)
    strategy.set_signals([TradeSignal("ABC", "BUY", 1)])
    runner._on_tick("ABC", 100.0)  # type: ignore[attr-defined]
    assert len(orders.orders) == 1
    strategy.set_signals([TradeSignal("ABC", "BUY", 1)])
    runner._on_tick("ABC", 100.0)  # type: ignore[attr-defined]
    assert len(orders.orders) == 1


def test_emergency_exit_liquidates_open_positions() -> None:
    runner, market_data, strategy, risk, orders, positions = create_runner(
        StrategyRunnerConfig(signal_cooldown=0.0)
    )
    open_positions: dict[str, Position] = {}
    runner._execute_signal(TradeSignal("ABC", "BUY", 1), 100.0, open_positions)  # type: ignore[attr-defined]
    market_data.set_price("ABC", 99.0)
    runner.trigger_emergency_exit()
    pos = positions.get_all_positions()[0]
    open_positions = {pos.symbol: pos}
    triggered = runner._check_exit_conditions("ABC", 99.0, pos, open_positions)  # type: ignore[attr-defined]
    assert triggered is True
    assert positions.get_all_positions() == []
