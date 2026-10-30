"""Portfolio budget screens use existing volatility payoff and fee models."""

import pytest

from nifty_scalper_bot.backtesting.volatility_research import (
    CondorAssessment,
    LongStraddleAssessment,
    screen_volatility_portfolio_budget,
)


@pytest.fixture(autouse=True)
def shadow(monkeypatch):
    monkeypatch.setenv("EXECUTION_MODE", "SHADOW")


@pytest.fixture
def condor():
    return CondorAssessment(
        credit_per_unit=43.0,
        gross_max_profit_rupees=2795.0,
        gross_max_loss_rupees=3705.0,
        entry_fees_rupees=150.0,
        max_call_width=100,
        max_put_width=100,
    )


def case(assessment, **overrides):
    inputs = {
        "assessment": assessment,
        "broker_required_margin": 15000.0,
        "broker_free_margin": 30000.0,
        "remaining_trade_risk": 5000.0,
        "remaining_daily_risk": 8000.0,
        "exit_settlement_fee_reserve": 200.0,
        "stressed_slippage_reserve": 400.0,
    }
    inputs.update(overrides)
    return screen_volatility_portfolio_budget(**inputs)


def test_condor_cost_and_stress_reserves_fit_both_limits(condor):
    result = case(condor)
    assert result.research_budget_fits is True
    assert result.planning_loss_reserve_rupees == pytest.approx(4455.0)
    assert result.planning_margin_reserve_rupees == pytest.approx(15600.0)
    assert result.live_eligible is False
    assert result.net_loss_bound_certified is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"broker_required_margin": None}, "research_budget_evidence_missing"),
        ({"broker_free_margin": float("nan")}, "research_budget_evidence_missing"),
        ({"remaining_daily_risk": None}, "research_budget_evidence_missing"),
        ({"exit_settlement_fee_reserve": 0}, "research_stress_or_margin_unverified"),
        ({"stressed_slippage_reserve": 0}, "research_stress_or_margin_unverified"),
        ({"broker_required_margin": 0}, "research_stress_or_margin_unverified"),
        ({"remaining_trade_risk": 4454}, "research_risk_budget_insufficient"),
        ({"remaining_daily_risk": 4400}, "research_risk_budget_insufficient"),
        ({"broker_free_margin": 15599}, "research_free_margin_insufficient"),
        ({"broker_free_margin": -1}, "research_budget_evidence_missing"),
    ],
)
def test_budget_missing_or_insufficient_blocks(condor, overrides, reason):
    result = case(condor, **overrides)
    assert result.research_budget_fits is False
    assert result.reason == reason
    assert result.live_eligible is False


def test_debit_straddle_uses_premium_at_risk_without_double_counting_entry_fees():
    assessment = LongStraddleAssessment(
        debit_per_unit=135.0,
        lower_expiry_breakeven=23865.0,
        upper_expiry_breakeven=24135.0,
        premium_plus_entry_fees_rupees=9000.0,
        forecast_minus_implied_vol=0.05,
    )
    result = case(
        assessment,
        broker_required_margin=9500.0,
        broker_free_margin=15000.0,
        remaining_trade_risk=10000.0,
        remaining_daily_risk=12000.0,
    )
    assert result.research_budget_fits is True
    assert result.planning_loss_reserve_rupees == pytest.approx(9600.0)
    assert result.planning_margin_reserve_rupees == pytest.approx(10100.0)


@pytest.mark.parametrize("mode", ["LIVE", "PAPER", ""])
def test_live_mode_cannot_authorize_portfolio_from_research_screen(
    condor, monkeypatch, mode
):
    monkeypatch.setenv("EXECUTION_MODE", mode)
    with pytest.raises(ValueError, match="volatility_research_requires_shadow_process"):
        case(condor)
