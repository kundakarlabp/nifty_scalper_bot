"""Tests for the option-native candidate scorer."""

from __future__ import annotations

from nifty_scalper_bot.strategies.option_signal import (
    _PRIOR_OI,
    _PRIOR_PRICE,
    score_option_candidate,
)


def setup_function() -> None:
    _PRIOR_OI.clear()
    _PRIOR_PRICE.clear()


def test_missing_metrics_is_neutral() -> None:
    delta, reasons = score_option_candidate("NIFTY26JUN25000CE", None)
    assert delta == 0.0 and reasons == ["option_metrics_unavailable"]


def test_rich_iv_penalized_cheap_iv_rewarded() -> None:
    delta, reasons = score_option_candidate("SYM1CE", {"iv": 0.75})
    assert delta == -1.0 and any("iv_rich" in r for r in reasons)
    delta, reasons = score_option_candidate("SYM2CE", {"iv": 0.22})
    assert delta == 0.5 and any("iv_reasonable" in r for r in reasons)


def test_oi_buildup_requires_premium_confirmation() -> None:
    d1, _ = score_option_candidate("SYM3CE", {"oi": 100000, "ltp": 100.0})
    assert d1 == 0.0

    d2, reasons = score_option_candidate("SYM3CE", {"oi": 105000, "ltp": 102.0})
    assert d2 == 0.5
    assert "oi_buildup_price_confirmed" in reasons

    d3, reasons = score_option_candidate("SYM3CE", {"oi": 110000, "ltp": 99.0})
    assert d3 == -0.5
    assert "oi_buildup_price_divergence" in reasons


def test_oi_buildup_ignores_sub_noise_premium_move(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_OI_MIN_PRICE_MOVE_PCT", "0.002")
    score_option_candidate("SYM_NOISE", {"oi": 100000, "ltp": 100.0})
    delta, reasons = score_option_candidate("SYM_NOISE", {"oi": 105000, "ltp": 100.05})
    assert delta == 0.0
    assert "oi_buildup_price_noise" in reasons


def test_oi_buildup_uses_half_spread_as_noise_floor(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_OI_MIN_PRICE_MOVE_PCT", "0.001")
    score_option_candidate("SYM_SPREAD", {"oi": 100000, "ltp": 100.0, "bid": 99.0, "ask": 101.0})
    delta, reasons = score_option_candidate("SYM_SPREAD", {"oi": 105000, "ltp": 100.5, "bid": 99.5, "ask": 101.5})
    assert delta == 0.0
    assert "oi_buildup_price_noise" in reasons


def test_oi_change_without_price_context_is_neutral() -> None:
    score_option_candidate("SYM_OI_ONLY", {"oi": 100000})
    delta, reasons = score_option_candidate("SYM_OI_ONLY", {"oi": 105000})
    assert delta == 0.0
    assert "oi_change_unconfirmed" in reasons


def test_oi_unwinding_is_not_directional_without_price_confirmation() -> None:
    score_option_candidate("SYM_UNWIND", {"oi": 100000, "ltp": 100.0})
    delta, reasons = score_option_candidate("SYM_UNWIND", {"oi": 95000, "ltp": 101.0})
    assert delta == 0.0
    assert "oi_unwinding_context" in reasons


def test_depth_imbalance() -> None:
    depth = {"buy": [{"quantity": 3000}], "sell": [{"quantity": 1000}]}
    delta, reasons = score_option_candidate("SYM4CE", {"depth": depth})
    assert delta == 0.5 and "depth_buy_support" in reasons
    depth = {"buy": [{"quantity": 1000}], "sell": [{"quantity": 3000}]}
    delta, reasons = score_option_candidate("SYM5CE", {"depth": depth})
    assert delta == -0.5 and "depth_sell_pressure" in reasons


def test_delta_clamped_and_disable_env(monkeypatch) -> None:
    score_option_candidate("SYM6CE", {"oi": 100000, "ltp": 100.0})
    delta, _ = score_option_candidate(
        "SYM6CE",
        {"iv": 0.20, "oi": 110000, "ltp": 102.0,
         "depth": {"buy": [{"quantity": 5000}], "sell": [{"quantity": 1000}]}},
    )
    assert delta == 1.5  # 0.5+0.5+0.5 clamped at +1.5
    monkeypatch.setenv("OPTION_SIGNAL_ENABLED", "false")
    delta, reasons = score_option_candidate("SYM6CE", {"iv": 0.9})
    assert delta == 0.0 and reasons == ["option_signal_disabled"]


def test_malformed_inputs_never_raise() -> None:
    delta, _ = score_option_candidate("SYM7CE", {"iv": "bad", "oi": object(), "ltp": object(), "depth": "junk"})
    assert delta == 0.0
