"""Unit tests for the premium decay strategy."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Sequence, Tuple

import pytest

from nifty_scalper_bot.execution.order_manager import OrderType
from nifty_scalper_bot.options.strike_selector import SelectedContract
from nifty_scalper_bot.strategies.premium_decay import PremiumDecayStrategy


class _StubOrderManager:
    """Order manager stub capturing routed payloads."""

    def __init__(self) -> None:
        self.placed: list[Mapping[str, Any]] = []

    def place_order(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        order_type: OrderType = OrderType.MARKET,
        price: float | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        product: str | None = None,
        tag: str | None = None,
        trailing_spec: Any | None = None,
        max_slippage_per_lot: float | None = None,
    ) -> str:
        """Capture the order request and return a mock identifier."""

        payload = {
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "order_type": order_type,
            "price": price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "product": product,
            "tag": tag,
            "trailing_spec": trailing_spec,
            "max_slippage_per_lot": max_slippage_per_lot,
        }
        self.placed.append(payload)
        return f"ORD-{len(self.placed)}"


class _StubRiskManager:
    """Risk manager stub allowing selective approvals."""

    def __init__(self, allow: bool = True) -> None:
        self.allow = allow
        self.calls: list[Mapping[str, Any]] = []

    def check_order(self, signal: Any, live_enabled: bool) -> tuple[bool, str]:
        """Record the signal and return configured approval."""

        del live_enabled
        self.calls.append(
            {
                "symbol": signal.symbol,
                "side": signal.side,
                "quantity": signal.quantity,
                "price": signal.price,
            }
        )
        return (self.allow, "")


class _StubOrchestrator:
    """Orchestrator stub forwarding signals without filtering."""

    def __init__(self) -> None:
        self.submissions: list[tuple[Any, str]] = []
        self.exits: list[str] = []

    def filter_signal(
        self, signal: Any, indicators: Mapping[str, Any], position_manager: Any
    ) -> Any | None:
        """Return the signal unchanged for testing."""

        del indicators, position_manager
        return signal

    def notify_submission(self, signal: Any, underlying: str) -> None:
        """Store the submission notification."""

        self.submissions.append((signal, underlying))

    def notify_exit(self, underlying: str) -> None:
        """Store the exit notification."""

        self.exits.append(underlying)


class _StubPositionManager:
    """Position manager stub returning no open positions."""

    def get_all_positions(self) -> Sequence[Any]:  # noqa: D401 - stub
        """Return an empty sequence of positions."""

        return []


def _make_contract(
    symbol: str, option_type: str, strike: float, delta: float
) -> SelectedContract:
    """Create a synthetic contract for strategy tests."""

    return SelectedContract(
        symbol=symbol,
        option_type=option_type,
        strike=strike,
        expiry=datetime.now(timezone.utc),
        ltp=50.0,
        delta=delta,
        metadata={"lot_size": 25},
    )


@pytest.fixture()
def dependencies() -> (
    Tuple[_StubOrderManager, _StubRiskManager, _StubOrchestrator, _StubPositionManager]
):
    """Return stubbed dependencies required by the strategy."""

    return (
        _StubOrderManager(),
        _StubRiskManager(),
        _StubOrchestrator(),
        _StubPositionManager(),
    )


def test_entry_places_both_legs(dependencies: tuple[Any, ...]) -> None:
    """Strategy should place two SELL orders when conditions are met."""

    order_manager, risk_manager, orchestrator, position_manager = dependencies
    strategy = PremiumDecayStrategy(
        order_manager=order_manager,
        risk_manager=risk_manager,
        orchestrator=orchestrator,
        position_manager=position_manager,
        lots=1,
        atr_threshold=30.0,
        adx_threshold=20.0,
    )
    indicators = {"atr": 12.0, "adx": 10.0}
    option_chain = [
        _make_contract("TESTCE", "CE", 20000, 0.26),
        _make_contract("TESTPE", "PE", 20000, -0.24),
    ]
    placed = strategy.evaluate_entry(
        underlying="NIFTY",
        indicators=indicators,
        option_chain=option_chain,
        iv=0.25,
    )
    assert placed is True
    assert len(order_manager.placed) == 2
    assert order_manager.placed[0]["side"] == "SELL"
    assert order_manager.placed[1]["side"] == "SELL"


def test_iv_spike_triggers_exit(dependencies: tuple[Any, ...]) -> None:
    """IV spike should force both legs to be bought back."""

    order_manager, risk_manager, orchestrator, position_manager = dependencies
    strategy = PremiumDecayStrategy(
        order_manager=order_manager,
        risk_manager=risk_manager,
        orchestrator=orchestrator,
        position_manager=position_manager,
        lots=1,
        atr_threshold=30.0,
        adx_threshold=20.0,
        iv_exit_threshold=0.30,
    )
    option_chain = [
        _make_contract("TESTCE", "CE", 20000, 0.26),
        _make_contract("TESTPE", "PE", 20000, -0.24),
    ]
    strategy.evaluate_entry(
        underlying="NIFTY",
        indicators={"atr": 10.0, "adx": 10.0},
        option_chain=option_chain,
        iv=0.25,
    )
    assert strategy._active is not None
    strategy.evaluate_exit(
        option_prices={"TESTCE": 55.0, "TESTPE": 53.0},
        current_iv=0.5,
        now=datetime.now(timezone.utc),
    )
    assert strategy._active is None
    assert len(orchestrator.exits) == 1
    assert len(order_manager.placed) >= 4
    assert order_manager.placed[-1]["side"] == "BUY"


def test_square_off_before_cutoff(dependencies: tuple[Any, ...]) -> None:
    """Square-off trigger should close open legs before end of day."""

    order_manager, risk_manager, orchestrator, position_manager = dependencies
    strategy = PremiumDecayStrategy(
        order_manager=order_manager,
        risk_manager=risk_manager,
        orchestrator=orchestrator,
        position_manager=position_manager,
        lots=1,
    )
    option_chain = [
        _make_contract("TESTCE", "CE", 20000, 0.26),
        _make_contract("TESTPE", "PE", 20000, -0.24),
    ]
    strategy.evaluate_entry(
        underlying="NIFTY",
        indicators={"atr": 8.0, "adx": 9.0},
        option_chain=option_chain,
        iv=0.2,
    )
    assert strategy._active is not None
    square_off_time = datetime(2024, 5, 1, 9, 50, tzinfo=timezone.utc)
    strategy.evaluate_exit(
        option_prices={"TESTCE": 48.0, "TESTPE": 49.0},
        current_iv=0.25,
        now=square_off_time,
    )
    assert strategy._active is None
    assert len(orchestrator.exits) == 1
