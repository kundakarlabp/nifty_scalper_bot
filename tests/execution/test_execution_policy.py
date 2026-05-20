from __future__ import annotations

from dataclasses import dataclass

import pytest

from nifty_scalper_bot.execution.execution_policy import ExecutionPolicy
from nifty_scalper_bot.utils.errors import OrderPlacementError


@dataclass
class DummyHub:
    quote: dict[str, float]

    def get_quote(self, symbol: str, allow_pull: bool = True) -> dict[str, float]:
        return dict(self.quote)


def test_build_plan_pegged_prices() -> None:
    hub = DummyHub({"best_bid": 100.0, "best_ask": 101.0})
    policy = ExecutionPolicy(
        hub, peg_k=0.25, retry_steps=3, timeout_sec=3.0, max_spread_pct=0.05
    )

    plan = policy.build_plan("TEST", "BUY")

    assert len(plan.limit_prices) == 3
    assert pytest.approx(plan.limit_prices[0], rel=1e-4) == 100.75
    assert pytest.approx(plan.limit_prices[1], rel=1e-4) == 101.0
    assert pytest.approx(plan.limit_prices[2], rel=1e-4) == 101.25
    assert plan.spread_pct > 0


def test_build_plan_rejects_wide_spread() -> None:
    hub = DummyHub({"best_bid": 100.0, "best_ask": 110.0})
    policy = ExecutionPolicy(
        hub, peg_k=0.1, retry_steps=2, timeout_sec=2.0, max_spread_pct=0.05
    )

    with pytest.raises(OrderPlacementError):
        policy.build_plan("TEST", "SELL")


def test_execution_policy_allows_configured_option_spread(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_ATM_PCT", "0.03")
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_LOW_PREMIUM_PCT", "0.06")
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_HARD_CAP_PCT", "0.10")
    monkeypatch.setenv("OPTION_LOW_PREMIUM_CUTOFF", "50")
    hub = DummyHub({"best_bid": 9.0, "best_ask": 9.5, "ltp": 10.0})
    policy = ExecutionPolicy(hub, max_spread_pct=0.015)
    plan = policy.build_plan("NFO:NIFTY26MAY23750CE", "BUY")
    assert plan.spread_pct > 0


def test_execution_policy_rejects_option_spread_above_hard_cap(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_ATM_PCT", "0.03")
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_LOW_PREMIUM_PCT", "0.06")
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_HARD_CAP_PCT", "0.10")
    monkeypatch.setenv("OPTION_LOW_PREMIUM_CUTOFF", "50")
    hub = DummyHub({"best_bid": 10.0, "best_ask": 12.0, "ltp": 10.0})
    policy = ExecutionPolicy(hub, max_spread_pct=0.015)
    with pytest.raises(OrderPlacementError):
        policy.build_plan("NFO:NIFTY26MAY23750CE", "BUY")


def test_execution_policy_keeps_strict_index_spread(monkeypatch) -> None:
    monkeypatch.setenv("OPTION_EXEC_MAX_SPREAD_ATM_PCT", "0.03")
    hub = DummyHub({"best_bid": 100.0, "best_ask": 103.0, "ltp": 101.0})
    policy = ExecutionPolicy(hub, max_spread_pct=0.015)
    with pytest.raises(OrderPlacementError):
        policy.build_plan("NSE:NIFTY", "BUY")
