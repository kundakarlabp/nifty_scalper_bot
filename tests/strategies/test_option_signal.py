"""Tests for the option-native candidate scorer."""

from __future__ import annotations

from nifty_scalper_bot.strategies.option_signal import (
    _PRIOR_OI,
    score_option_candidate,
)


def setup_function() -> None:
    _PRIOR_OI.clear()


def test_missing_metrics_is_neutral() -> None:
    delta, reasons = score_option_candidate("NIFTY26JUN25000CE", None)
    assert delta == 0.0 and reasons == ["option_metrics_unavailable"]


def test_rich_iv_penalized_cheap_iv_rewarded() -> None:
    delta, reasons = score_option_candidate("SYM1CE", {"iv": 0.75})
    assert delta == -1.0 and any("iv_rich" in r for r in reasons)
    delta, reasons = score_option_candidate("SYM2CE", {"iv": 0.22})
    assert delta == 0.5 and any("iv_reasonable" in r for r in reasons)


def test_oi_buildup_rewarded_on_second_observation() -> None:
    d1, _ = score_option_candidate("SYM3CE", {"oi": 100000})
    assert d1 == 0.0  # first sight: no prior
    d2, reasons = score_option_candidate("SYM3CE", {"oi": 105000})
    assert d2 == 0.5 and "oi_buildup" in reasons
    d3, reasons = score_option_candidate("SYM3CE", {"oi": 100000})
    assert d3 == -0.25 and "oi_unwinding" in reasons


def test_depth_imbalance() -> None:
    depth = {"buy": [{"quantity": 3000}], "sell": [{"quantity": 1000}]}
    delta, reasons = score_option_candidate("SYM4CE", {"depth": depth})
    assert delta == 0.5 and "depth_buy_support" in reasons
    depth = {"buy": [{"quantity": 1000}], "sell": [{"quantity": 3000}]}
    delta, reasons = score_option_candidate("SYM5CE", {"depth": depth})
    assert delta == -0.5 and "depth_sell_pressure" in reasons


def test_delta_clamped_and_disable_env(monkeypatch) -> None:
    score_option_candidate("SYM6CE", {"oi": 100000})
    delta, _ = score_option_candidate(
        "SYM6CE",
        {"iv": 0.20, "oi": 110000,
         "depth": {"buy": [{"quantity": 5000}], "sell": [{"quantity": 1000}]}},
    )
    assert delta == 1.5  # 0.5+0.5+0.5 clamped at +1.5
    monkeypatch.setenv("OPTION_SIGNAL_ENABLED", "false")
    delta, reasons = score_option_candidate("SYM6CE", {"iv": 0.9})
    assert delta == 0.0 and reasons == ["option_signal_disabled"]


def test_malformed_inputs_never_raise() -> None:
    delta, _ = score_option_candidate("SYM7CE", {"iv": "bad", "oi": object(), "depth": "junk"})
    assert delta == 0.0
