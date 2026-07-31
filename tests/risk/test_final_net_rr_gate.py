from __future__ import annotations

from types import SimpleNamespace

from nifty_scalper_bot.risk.entry_guard_patch import (
    _net_rr_block_reason,
    _real_broker_live,
)
from nifty_scalper_bot.risk.net_rr_gate import evaluate_final_net_rr


def _signal(*, target: float, stop: float = 92.0, **metadata):
    return SimpleNamespace(
        symbol="NFO:NIFTY2680625000CE",
        action="BUY",
        quantity=65,
        entry_price=100.0,
        stop_loss=stop,
        take_profit=target,
        metadata={"bid": 99.5, "ask": 100.5, **metadata},
    )


def test_final_net_rr_accepts_economic_trade(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")

    result = evaluate_final_net_rr(_signal(target=116.0))

    assert result is not None
    assert result.allowed is True
    assert result.net_rr >= 1.5
    assert result.net_reward < result.gross_reward
    assert result.net_risk > result.gross_risk


def test_final_net_rr_rejects_grossly_valid_but_net_weak_trade(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    signal = _signal(target=112.0)

    result = evaluate_final_net_rr(signal)
    blocked = _net_rr_block_reason(signal)

    assert result is not None
    assert (target_rr := (112.0 - 100.0) / (100.0 - 92.0)) == 1.5
    assert target_rr >= 1.5
    assert result.net_rr < 1.5
    assert blocked is not None
    assert "net reward-risk insufficient" in blocked[0]


def test_final_gate_uses_strategy_replaced_sl_and_target(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "1.5")
    signal = SimpleNamespace(
        symbol="NFO:NIFTY2680625000PE",
        action="BUY",
        quantity=65,
        stop_loss=92.0,
        take_profit=116.0,
        metadata={
            "premium_risk_reference_price": 100.0,
            "premium_risk_contract_applied": True,
            "bid": 99.5,
            "ask": 100.5,
        },
    )

    result = evaluate_final_net_rr(signal)

    assert result is not None
    assert result.allowed is True


def test_wider_spread_reduces_net_reward_risk(monkeypatch) -> None:
    monkeypatch.setenv("MIN_NET_REWARD_RISK", "0")
    tight = evaluate_final_net_rr(_signal(target=116.0, bid=99.9, ask=100.1))
    wide = evaluate_final_net_rr(_signal(target=116.0, bid=99.0, ask=101.0))

    assert tight is not None and wide is not None
    assert wide.net_rr < tight.net_rr
    assert wide.target_cost > tight.target_cost


def test_economic_rejection_is_real_broker_live_only(monkeypatch) -> None:
    for name in ("BROKER_SIMULATION", "PAPER_MODE", "PAPER__ENABLED", "SHADOW_MODE"):
        monkeypatch.delenv(name, raising=False)
    assert _real_broker_live(True) is True
    assert _real_broker_live(False) is False

    monkeypatch.setenv("BROKER_SIMULATION", "true")
    assert _real_broker_live(True) is False
    monkeypatch.delenv("BROKER_SIMULATION")

    monkeypatch.setenv("PAPER_MODE", "true")
    assert _real_broker_live(True) is False


def test_non_option_and_exit_orders_are_not_gated() -> None:
    non_option = SimpleNamespace(
        symbol="NSE:NIFTY",
        action="BUY",
        quantity=65,
        entry_price=25000.0,
        stop_loss=24950.0,
        take_profit=25100.0,
        metadata={},
    )
    exit_order = SimpleNamespace(
        symbol="NFO:NIFTY2680625000CE",
        action="SELL",
        quantity=65,
        entry_price=100.0,
        stop_loss=92.0,
        take_profit=116.0,
        metadata={},
    )

    assert evaluate_final_net_rr(non_option) is None
    assert evaluate_final_net_rr(exit_order) is None
    assert _net_rr_block_reason(exit_order) is None
