from __future__ import annotations

import time

import pytest

from src.risk.guards import RiskConfig, RiskGuards


def test_rate_limit_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_TRADING", "true")
    cfg = RiskConfig(max_orders_per_min=2, trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    assert guards.ok_to_trade()
    assert guards.ok_to_trade()
    assert not guards.ok_to_trade()


def test_daily_loss_cap() -> None:
    cfg = RiskConfig(daily_loss_cap=100, trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    guards.set_pnl_today(-150)
    assert not guards.ok_to_trade()


def test_kill_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = RiskConfig(kill_env="TEST_KILL", trading_start_hm="00:00", trading_end_hm="23:59")
    guards = RiskGuards(cfg)
    monkeypatch.setenv("TEST_KILL", "false")
    try:
        assert not guards.ok_to_trade()
    finally:
        monkeypatch.delenv("TEST_KILL", raising=False)


def test_trading_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ENABLE_TRADING", "true")
    cfg = RiskConfig(trading_start_hm="10:00", trading_end_hm="10:01")
    guards = RiskGuards(cfg)

    def fake_strftime(fmt: str, ts: float) -> str:  # pragma: no cover - simple stub
        return "09:00"

    monkeypatch.setattr(time, "strftime", fake_strftime)
    assert not guards.ok_to_trade()

    monkeypatch.setattr(time, "strftime", lambda fmt, ts: "10:00")
    assert guards.ok_to_trade()
