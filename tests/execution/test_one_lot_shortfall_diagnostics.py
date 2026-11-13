"""Realistic one-lot cash/risk shortfall diagnostics; never order authority."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.affordability import (
    evaluate_minimum_lot_affordability,
)


def _manager(*, authoritative_snapshot: bool = False) -> SimpleNamespace:
    risk = SimpleNamespace(
        account_balance=15_466.0,
        settings=SimpleNamespace(per_trade_risk_pct=5.0),
        _switches=SimpleNamespace(
            max_day_loss=773.30, day_loss=lambda: 0.0
        ),
    )
    if authoritative_snapshot:
        risk.snapshot = lambda: None  # fail closed; no stale switch fallback
    return SimpleNamespace(
        _margin_factor=1.1,
        _margin_buffer=0.9,
        _risk_manager=risk,
        resolve_lot_size=lambda _symbol: 65,
    )


def _decision(
    symbol: str,
    *,
    bid: float,
    ask: float,
    stop: float,
    unavailable_snapshot: bool = False,
):
    return evaluate_minimum_lot_affordability(
        symbol=symbol,
        quote={"bid": bid, "ask": ask, "tick_size": 0.05},
        order_manager=_manager(authoritative_snapshot=unavailable_snapshot),
        data_hub=SimpleNamespace(
            get_available_balance=lambda force=False: 15_466.0
        ),
        fallback_balance=15_466.0,
        plan_entry_price=ask,
        plan_stop_loss=stop,
    )


def test_12_nov_cash_and_risk_shortfalls_are_quantified_not_waived() -> None:
    # Same quote, lot and executable cash as the production VWAP CE rejection.
    result = _decision(
        "NFO:NIFTY26N1725300CE",
        bid=219.95,
        ask=220.45,
        stop=206.45,
    )
    assert result.required == pytest.approx(14_329.25)
    assert result.executable_capacity == pytest.approx(13_919.40)
    assert result.cash_shortfall == pytest.approx(409.85)
    assert result.plan_risk_shortfall is not None
    assert result.plan_risk_shortfall > 0.0
    assert result.affordable is False
    assert result.reason == "minimum_lot_unaffordable"
    assert result.capacity_blocker == "cash"
    assert result.to_dict()["cash_shortfall"] == pytest.approx(409.85)


def test_cash_viable_option_can_still_fail_preserved_stop_risk() -> None:
    result = _decision(
        "NFO:NIFTY26N1725350CE",
        bid=195.95,
        ask=196.50,
        stop=182.50,
    )
    assert result.cash_shortfall == pytest.approx(0.0)
    assert result.cash_affordable is True
    assert result.plan_risk_shortfall is not None
    assert result.plan_risk_shortfall > 0.0
    assert result.reason == "minimum_lot_risk_unaffordable"
    assert result.affordable is False
    assert result.effective_one_lot_risk_budget == pytest.approx(773.30)


def test_missing_authoritative_risk_snapshot_does_not_claim_zero_shortfall() -> None:
    result = _decision(
        "NFO:NIFTY26N1725350CE",
        bid=195.95,
        ask=196.50,
        stop=182.50,
        unavailable_snapshot=True,
    )
    assert result.cash_shortfall == pytest.approx(0.0)
    assert result.plan_risk_shortfall is None
    assert result.reason == "risk_budget_unavailable"
    assert result.affordable is False
    assert result.day_loss_source == "risk_snapshot_unavailable"
