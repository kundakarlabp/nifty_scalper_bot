from __future__ import annotations

from types import SimpleNamespace

import pytest

from nifty_scalper_bot.execution.runtime_order_manager import _cost_adjust_entry_target
from nifty_scalper_bot.risk.net_rr_gate import (
    evaluate_final_net_rr,
    minimum_target_for_net_rr,
)


SYMBOL = "NFO:NIFTY2681824200CE"


def _signal(*, target: float, entry: float = 29.45, stop: float = 26.21):
    return SimpleNamespace(
        symbol=SYMBOL,
        action="BUY",
        quantity=65,
        entry_price=entry,
        stop_loss=stop,
        take_profit=target,
        metadata={"bid": 29.40, "ask": 29.50},
    )


def test_minimum_target_repairs_20260818_cost_eroded_two_r_case(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    signal = _signal(target=35.94)

    before = evaluate_final_net_rr(signal)
    target = minimum_target_for_net_rr(signal)

    assert before is not None and before.allowed is False
    assert before.net_rr == pytest.approx(1.3515, rel=1e-3)
    assert target is not None
    assert target > signal.take_profit
    # The target must remain a modest bounded uplift, not an arbitrary expansion.
    risk_points = signal.entry_price - signal.stop_loss
    assert (target - signal.entry_price) / risk_points <= 2.35

    repaired = _signal(target=target)
    after = evaluate_final_net_rr(repaired)
    assert after is not None and after.allowed is True
    assert after.net_rr >= 1.5


def test_minimum_target_fails_closed_when_required_uplift_exceeds_cap(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.05")
    signal = _signal(target=35.94)

    assert minimum_target_for_net_rr(signal) is None


def test_runtime_adjusts_only_distance_anchored_entry_and_preserves_risk(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    monkeypatch.setenv("MAX_NET_RR_TARGET_UPLIFT_R", "0.35")
    manager = SimpleNamespace(
        _get_latest_quote_safe=lambda _symbol: {"bid": 29.40, "ask": 29.50},
        _extract_quote_diagnostics=lambda quote: quote,
        _logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    kwargs = {
        "symbol": SYMBOL,
        "side": "BUY",
        "quantity": 65,
        "price": 29.45,
        "stop_loss": 26.21,
        "take_profit": 35.94,
        "check_risk": True,
        "intent": "ENTRY",
        "trade_provenance": {
            "bracket_anchor_mode": "distance",
            "initial_reward_risk": 2.0,
        },
    }

    adjusted = _cost_adjust_entry_target(manager, kwargs)

    assert adjusted is not kwargs
    assert adjusted["price"] == kwargs["price"]
    assert adjusted["stop_loss"] == kwargs["stop_loss"]
    assert adjusted["quantity"] == kwargs["quantity"]
    assert adjusted["take_profit"] > kwargs["take_profit"]
    assert adjusted["trade_provenance"]["net_rr_target_adjusted"] is True
    assert adjusted["trade_provenance"]["original_take_profit"] == kwargs["take_profit"]


def test_runtime_never_moves_absolute_strategy_target(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    manager = SimpleNamespace(
        _get_latest_quote_safe=lambda _symbol: {"bid": 29.40, "ask": 29.50},
        _extract_quote_diagnostics=lambda quote: quote,
        _logger=SimpleNamespace(info=lambda *args, **kwargs: None),
    )
    kwargs = {
        "symbol": SYMBOL,
        "side": "BUY",
        "quantity": 65,
        "price": 29.45,
        "stop_loss": 26.21,
        "take_profit": 35.94,
        "check_risk": True,
        "intent": "ENTRY",
        "trade_provenance": {"bracket_anchor_mode": "absolute_level"},
    }

    adjusted = _cost_adjust_entry_target(manager, kwargs)

    assert adjusted == kwargs
