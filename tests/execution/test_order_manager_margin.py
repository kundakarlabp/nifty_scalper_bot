from __future__ import annotations

import types
from pathlib import Path

import pytest

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.execution.margin_engine import MarginDecision, SizingResult
from nifty_scalper_bot.execution.order_manager import OrderManager, OrderType
from nifty_scalper_bot.execution.position_manager import PositionManager
from nifty_scalper_bot.utils.errors import OrderPlacementError
from nifty_scalper_bot.utils.rate_limiter import RateLimiter
from nifty_scalper_bot.utils.reasons import canonical


class DummyBroker:
    def place_order(self, payload: dict[str, object]) -> dict[str, object]:
        return {"order_id": "ORD-1", "status": "open"}


class DummyRisk:
    def __init__(self, balance: float) -> None:
        self.current_balance = balance
        self.settings = RiskSettings(
            per_trade_risk_pct=100.0,
            per_trade_cap_pct=1000.0,
        )

    def risk_gate_should_trade(self) -> tuple[bool, list[str]]:
        return True, []

    def cooldown_remaining(self) -> float:
        return 0.0


@pytest.fixture()
def order_manager(tmp_path: Path) -> OrderManager:
    limiter = RateLimiter()
    limiter.configure_bucket("orders", capacity=5, refill_rate_per_sec=5)
    positions = PositionManager(state_file=str(tmp_path / "positions.json"))
    manager = OrderManager(
        DummyBroker(),
        positions,
        limiter,
        history_path=tmp_path / "history.json",
    )

    class Resolver:
        @staticmethod
        def lot_size_for_symbol(symbol: str) -> int:
            return 1

    manager.set_instrument_resolver(Resolver())
    manager.set_risk_manager(DummyRisk(balance=1_000.0))
    manager._margin_factor = 1.0
    manager._margin_buffer = 1.0
    manager._reference_price = types.MethodType(lambda self, _: 100.0, manager)
    manager._execution_policy = None
    return manager


def test_pretrade_soft_block_no_raise_margin(order_manager: OrderManager) -> None:
    order_manager.set_risk_manager(DummyRisk(balance=500.0))
    order_manager._margin_buffer = 0.9
    with pytest.raises(OrderPlacementError) as exc_info:
        order_manager._execute_market_order(
            symbol="NIFTY",
            side="BUY",
            quantity=10,
            product="NRML",
            tag=None,
        )
    assert canonical(str(exc_info.value)) == "margin_no_qty"


def test_broker_400_insufficient_maps_to_soft_margin(
    order_manager: OrderManager,
) -> None:
    order_manager.set_risk_manager(DummyRisk(balance=1_000_000.0))
    order_manager._margin_buffer = 0.5
    plan = types.SimpleNamespace(limit_prices=[5.0], step_timeout=0.01, spread_pct=0.0)
    order_manager._execution_policy = types.SimpleNamespace(
        build_plan=lambda symbol, side: plan,
        timeout_sec=0.5,
    )

    def _raise_margin(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        order_type: OrderType,
        price: float | None,
        product: str | None,
        tag: str | None,
    ) -> object:
        raise OrderPlacementError("Insufficient funds available")

    order_manager._place_single_order = types.MethodType(_raise_margin, order_manager)

    with pytest.raises(OrderPlacementError) as exc_info:
        order_manager._execute_market_order(
            symbol="NIFTY",
            side="BUY",
            quantity=1,
            product="NRML",
            tag="test",
        )
    assert canonical(str(exc_info.value)) == "MARGIN"


def test_broker_400_mis_window_maps_soft(order_manager: OrderManager) -> None:
    order_manager.set_risk_manager(DummyRisk(balance=1_000_000.0))
    order_manager._margin_buffer = 0.5
    plan = types.SimpleNamespace(limit_prices=[5.0], step_timeout=0.01, spread_pct=0.0)
    order_manager._execution_policy = types.SimpleNamespace(
        build_plan=lambda symbol, side: plan,
        timeout_sec=0.5,
    )

    def _raise_mis(
        self,
        *,
        symbol: str,
        side: str,
        quantity: int,
        order_type: OrderType,
        price: float | None,
        product: str | None,
        tag: str | None,
    ) -> object:
        raise OrderPlacementError("Intraday orders (MIS) are allowed only till 3:25 PM")

    order_manager._place_single_order = types.MethodType(_raise_mis, order_manager)

    with pytest.raises(OrderPlacementError) as exc_info:
        order_manager._execute_market_order(
            symbol="NIFTY",
            side="BUY",
            quantity=1,
            product="MIS",
            tag="test",
        )
    assert canonical(str(exc_info.value)) == "MIS_WINDOW_CLOSED"


def test_margin_block_cooldown_engages(order_manager: OrderManager) -> None:
    sizing = SizingResult(
        qty=0,
        reason="insufficient_margin",
        needed=5_000.0,
        available=1_000.0,
    )
    decision = MarginDecision(
        ok=False,
        reason="MARGIN insufficient_margin",
        order_type="NRML",
        quantity=0,
        est_required=5_000.0,
        available=1_000.0,
        sizing=sizing,
    )
    context = {"skip_reason": "margin_no_qty"}
    for _ in range(3):
        order_manager._register_margin_block(
            symbol="NIFTY",
            side="BUY",
            quantity=75,
            reason="margin_no_qty",
            decision=decision,
            context=context,
        )
    active, remaining = order_manager._margin_cooldown_state()
    assert active
    assert remaining > 0
    with pytest.raises(OrderPlacementError) as exc_info:
        order_manager._execute_market_order(
            symbol="NIFTY",
            side="BUY",
            quantity=1,
            product="NRML",
            tag=None,
        )
    assert str(exc_info.value).strip().lower() == "margin_cooldown"
