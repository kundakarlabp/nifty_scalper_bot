from __future__ import annotations

from nifty_scalper_bot.execution.premium_risk_contract_patch import (
    apply_premium_risk_contract,
)
from nifty_scalper_bot.strategies.signal_generator import Signal


def _signal(*, action: str = "BUY", stop_loss=None, take_profit=None, **metadata):
    return Signal(
        action=action,
        symbol="NFO:NIFTY2680424400PE",
        quantity=65,
        confidence=0.8,
        reason="setup",
        stop_loss=stop_loss,
        take_profit=take_profit,
        metadata=metadata,
    )


def test_absolute_premium_distance_builds_buy_geometry() -> None:
    signal = _signal(
        premium_stop_distance=8.0,
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
    )

    result = apply_premium_risk_contract(signal, 100.0)

    assert result.stop_loss == 92.0
    assert result.take_profit == 116.0
    assert result.metadata["premium_risk_source"] == "premium_stop_distance"


def test_absolute_distance_is_not_replaced_by_legacy_percentage() -> None:
    signal = _signal(
        premium_stop_distance=8.0,
        premium_stop_pct=0.02,
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
    )

    result = apply_premium_risk_contract(signal, 100.0)

    assert result.stop_loss == 92.0
    assert result.take_profit == 116.0


def test_existing_valid_strategy_levels_are_preserved() -> None:
    signal = _signal(
        stop_loss=90.0,
        take_profit=125.0,
        premium_stop_distance=8.0,
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
    )

    result = apply_premium_risk_contract(signal, 100.0)

    assert result.stop_loss == 90.0
    assert result.take_profit == 125.0


def test_non_premium_domain_is_not_modified() -> None:
    signal = _signal(
        premium_stop_distance=8.0,
        premium_target_rr=2.0,
        invalidation_level_domain="underlying",
    )

    result = apply_premium_risk_contract(signal, 100.0)

    assert result is signal
    assert result.stop_loss is None


def test_sell_geometry_is_symmetric() -> None:
    signal = _signal(
        action="SELL",
        premium_stop_distance=5.0,
        premium_target_rr=1.5,
        invalidation_level_domain="option_premium",
    )

    result = apply_premium_risk_contract(signal, 100.0)

    assert result.stop_loss == 105.0
    assert result.take_profit == 92.5
