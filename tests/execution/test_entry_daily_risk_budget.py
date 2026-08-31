from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.affordability import evaluate_minimum_lot_affordability
from nifty_scalper_bot.execution.margin_engine import MarginEngine
from nifty_scalper_bot.execution.order_manager_core import OrderManager, TradePlan


class _Switches:
    def __init__(self, max_day_loss: float, day_loss: float = 0.0) -> None:
        self.max_day_loss = max_day_loss
        self._day_loss = day_loss

    def day_loss(self) -> float:
        return self._day_loss


def _order_manager(*, max_day_loss: float = 319.09, day_loss: float = 0.0):
    manager = OrderManager.__new__(OrderManager)
    manager._risk_manager = SimpleNamespace(
        account_balance=15_954.60,
        settings=SimpleNamespace(
            per_trade_risk_pct=5.0,
            per_trade_cap_pct=100.0,
            min_lots_per_trade=1,
            max_lots_per_trade=1,
            atr_stop_multiple=1.0,
        ),
        _switches=_Switches(max_day_loss, day_loss),
    )
    manager._margin_engine = MarginEngine(
        broker=object(), data_hub=None, lot_size_resolver=None, clock=lambda: 0.0
    )
    manager._margin_factor = 1.0
    manager._margin_buffer = 1.0
    manager._logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
    )
    manager.resolve_lot_size = lambda _symbol: 65
    return manager


def _plan(entry: float, stop: float) -> TradePlan:
    return TradePlan(
        symbol="NFO:NIFTY2690124050PE",
        side="BUY",
        quantity=65,
        entry_price=entry,
        stop_loss=stop,
        take_profit=entry + 2.0 * (entry - stop),
        product="NRML",
        requested_lots=1,
        resolved_lot_size=65,
        trace_id="daily-risk-regression",
    )


def test_entry_sizing_rejects_production_one_lot_risk_above_remaining_day_budget():
    manager = _order_manager(max_day_loss=319.09)
    plan = _plan(64.65, 59.70)  # 4.95 * 65 = 321.75

    decision = OrderManager._plan_entry_margin(
        manager,
        plan=plan,
        price=64.65,
        lot_size=65,
        available_balance=15_954.60,
    )

    assert not decision.ok
    assert decision.quantity == 0
    assert decision.reason == "MARGIN no_qty_after_risk"


def test_entry_sizing_accepts_one_lot_when_stop_risk_fits_remaining_day_budget():
    manager = _order_manager(max_day_loss=319.09)
    plan = _plan(65.30, 60.60)  # 4.70 * 65 = 305.50

    decision = OrderManager._plan_entry_margin(
        manager,
        plan=plan,
        price=65.30,
        lot_size=65,
        available_balance=15_954.60,
    )

    assert decision.ok
    assert decision.quantity == 65


def test_entry_sizing_accounts_for_already_consumed_day_loss_capacity():
    manager = _order_manager(max_day_loss=319.09, day_loss=100.0)
    plan = _plan(65.30, 60.60)  # 305.50 > 219.09 remaining

    decision = OrderManager._plan_entry_margin(
        manager,
        plan=plan,
        price=65.30,
        lot_size=65,
        available_balance=15_954.60,
    )

    assert not decision.ok
    assert decision.quantity == 0


def test_no_configured_daily_cap_preserves_existing_per_trade_risk_policy():
    manager = _order_manager(max_day_loss=0.0)
    plan = _plan(64.65, 59.70)

    decision = OrderManager._plan_entry_margin(
        manager,
        plan=plan,
        price=64.65,
        lot_size=65,
        available_balance=15_954.60,
    )

    assert decision.ok
    assert decision.quantity == 65


def test_affordability_telemetry_exposes_risk_capacity_without_changing_cash_readiness():
    manager = _order_manager(max_day_loss=319.09)
    manager._margin_factor = 1.1
    manager._margin_buffer = 0.9

    decision = evaluate_minimum_lot_affordability(
        symbol="NFO:NIFTY2690124050PE",
        quote={"ask": 65.30},
        order_manager=manager,
        fallback_balance=15_954.60,
    )

    assert decision.affordable is True
    assert decision.remaining_daily_risk_budget == pytest.approx(319.09)
    assert decision.per_trade_risk_budget == pytest.approx(797.73)
    assert decision.effective_one_lot_risk_budget == pytest.approx(319.09)
    assert decision.max_stop_distance_one_lot == pytest.approx(319.09 / 65.0)
