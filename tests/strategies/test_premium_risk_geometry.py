from __future__ import annotations

import inspect
import subprocess
import sys

import pytest

from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr
from nifty_scalper_bot.strategies.premium_risk_geometry import (
    anchor_option_geometry_to_execution,
    apply_cost_aware_risk_floor,
    apply_premium_risk_contract,
    validate_option_premium_geometry,
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


def test_cost_floor_repairs_target_without_widening_stop(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    signal = apply_premium_risk_contract(
        _signal(
            premium_stop_distance=1.2,
            premium_target_rr=1.8,
            invalidation_level_domain="option_premium",
            bracket_anchor_mode="distance",
            bid=59.9,
            ask=60.1,
        ),
        60.0,
    )

    original_stop = signal.stop_loss
    original_target = signal.take_profit
    result = apply_cost_aware_risk_floor(
        signal,
        entry_price=60.0,
        quantity=65,
        half_spread=0.1,
    )

    assert result.stop_loss == pytest.approx(original_stop)
    assert 60.0 - result.stop_loss == pytest.approx(1.2)
    assert result.take_profit >= original_target
    assert result.metadata["premium_cost_floor_applied"] is False
    assert result.metadata["premium_cost_floor_original_distance"] == pytest.approx(1.2)
    # This very narrow 1.8R setup may be uneconomic even after the bounded
    # target uplift. The critical invariant is that transaction costs never
    # increase its stop-loss exposure.
    assert result.metadata["premium_cost_floor_distance"] > 1.2


def test_live_2pct_regression_target_uplift_preserves_stop_and_clears_net_rr(monkeypatch) -> None:
    """10-Sep live economics: costs must not turn a safe stop into >2% risk."""
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    entry = 82.15
    distance = 4.50
    signal = _signal(
        stop_loss=entry - distance,
        take_profit=entry + 2.0 * distance,
        premium_stop_distance=distance,
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
        bracket_anchor_mode="distance",
        entry_price=entry,
        bid=81.95,
        ask=82.15,
    )

    before_risk = (entry - signal.stop_loss) * 65
    result = apply_cost_aware_risk_floor(
        signal,
        entry_price=entry,
        quantity=65,
        half_spread=0.10,
    )
    after_risk = (entry - result.stop_loss) * 65

    assert before_risk == pytest.approx(292.50)
    assert after_risk == pytest.approx(before_risk)
    assert result.stop_loss == pytest.approx(signal.stop_loss)
    assert result.take_profit >= signal.take_profit
    net_rr = evaluate_final_net_rr(result)
    assert net_rr is not None and net_rr.allowed is True
    assert net_rr.net_rr >= 1.5


def test_cost_floor_does_not_move_absolute_technical_geometry() -> None:
    signal = _signal(
        stop_loss=55.0,
        take_profit=70.0,
        bracket_anchor_mode="absolute_level",
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
    )

    result = apply_cost_aware_risk_floor(
        signal,
        entry_price=60.0,
        quantity=65,
        half_spread=0.0,
    )

    assert result is signal


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


def test_untrusted_spot_scale_atr_is_not_used_as_premium_distance() -> None:
    signal = _signal(stop_loss=82.5, take_profit=195.0)

    result = validate_option_premium_geometry(
        None,
        signal,
        entry_price=120.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == 108.0
    assert result.take_profit == 144.0
    assert result.metadata["premium_risk_source"] == "premium_percent_fallback"
    assert result.metadata["premium_risk_domain"] == "option_premium"


def test_underlying_scale_target_is_rebuilt_symmetrically() -> None:
    signal = _signal(stop_loss=110.0, take_profit=24900.0)

    result = validate_option_premium_geometry(
        None,
        signal,
        entry_price=120.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == 110.0
    assert result.take_profit == 140.0
    assert (result.take_profit - 120.0) / (120.0 - result.stop_loss) == 2.0


def test_premium_target_rr_remains_authoritative() -> None:
    signal = _signal(
        stop_loss=92.0,
        take_profit=150.0,
        premium_stop_distance=8.0,
        premium_target_rr=1.5,
        invalidation_level_domain="option_premium",
    )

    result = validate_option_premium_geometry(
        None,
        signal,
        entry_price=100.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == 92.0
    assert result.take_profit == 112.0


def test_replacement_candidate_geometry_beats_stale_original_distance() -> None:
    signal = _signal(
        stop_loss=112.746,
        take_profit=138.2364,
        candidate_selected=True,
        candidate_symbol="NFO:NIFTY2680424400PE",
        candidate_entry_price=122.55,
        candidate_stop_loss=112.746,
        candidate_target=138.2364,
        premium_stop_distance=2.8457142857,
        premium_target_rr=2.0,
        premium_risk_reference_price=92.65,
        invalidation_level_domain="option_premium",
    )

    result = validate_option_premium_geometry(
        None,
        signal,
        entry_price=122.55,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == pytest.approx(112.746)
    assert result.take_profit == pytest.approx(142.158)
    assert result.metadata["premium_risk_source"] == "selected_candidate_geometry"
    rr = (result.take_profit - 122.55) / (122.55 - result.stop_loss)
    assert rr == pytest.approx(2.0)


def test_candidate_metadata_does_not_override_uncopied_strategy_geometry() -> None:
    signal = _signal(
        stop_loss=119.7042857143,
        take_profit=128.2414285714,
        candidate_selected=True,
        candidate_symbol="NFO:NIFTY2680424400PE",
        candidate_entry_price=122.55,
        candidate_stop_loss=112.746,
        candidate_target=138.2364,
        premium_stop_distance=2.8457142857,
        premium_target_rr=2.0,
        invalidation_level_domain="option_premium",
    )

    result = validate_option_premium_geometry(
        None,
        signal,
        entry_price=122.55,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == pytest.approx(122.55 - 2.8457142857)
    assert result.metadata["premium_risk_source"] == "premium_stop_distance"


def test_execution_anchor_preserves_valid_geometry_instead_of_widening() -> None:
    signal = _signal(stop_loss=92.0, take_profit=116.0)

    result = anchor_option_geometry_to_execution(
        None,
        signal,
        signal_price=100.0,
        execution_price=100.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == 92.0
    assert result.take_profit == 116.0


def test_execution_anchor_translates_distance_geometry() -> None:
    signal = _signal(stop_loss=92.0, take_profit=116.0)

    result = anchor_option_geometry_to_execution(
        None,
        signal,
        signal_price=100.0,
        execution_price=105.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result.stop_loss == 97.0
    assert result.take_profit == 121.0


def test_absolute_invalidation_is_not_moved_by_anchor() -> None:
    signal = _signal(
        stop_loss=92.0,
        take_profit=116.0,
        bracket_anchor_mode="absolute_level",
    )

    result = anchor_option_geometry_to_execution(
        None,
        signal,
        signal_price=100.0,
        execution_price=105.0,
        entry_side="BUY",
        atr=25.0,
    )

    assert result is signal
    assert result.stop_loss == 92.0
    assert result.take_profit == 116.0


def test_runner_geometry_is_owned_at_definition_site() -> None:
    from nifty_scalper_bot.strategies.runner import StrategyRunner

    assert StrategyRunner._validate_long_option_geometry.__module__ == (
        "nifty_scalper_bot.strategies.runner"
    )
    assert StrategyRunner._anchor_sl_tp_to_execution.__module__ == (
        "nifty_scalper_bot.strategies.runner"
    )


def test_plain_runner_import_has_canonical_geometry_without_core_app() -> None:
    code = """
from nifty_scalper_bot.strategies.runner import StrategyRunner
print('VALIDATE=' + StrategyRunner._validate_long_option_geometry.__module__)
print('ANCHOR=' + StrategyRunner._anchor_sl_tp_to_execution.__module__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "VALIDATE=nifty_scalper_bot.strategies.runner" in result.stdout
    assert "ANCHOR=nifty_scalper_bot.strategies.runner" in result.stdout


def test_geometry_helper_has_no_class_mutation_hooks() -> None:
    import nifty_scalper_bot.strategies.premium_risk_geometry as geometry

    source = inspect.getsource(geometry)
    assert "StrategyRunner." not in source
    assert "StrategyManager.generate_signal =" not in source
    assert "install_runner_geometry_hardening" not in source
    assert "install_bracket_exit_provenance_hardening" not in source
