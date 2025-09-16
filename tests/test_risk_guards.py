from __future__ import annotations

import time

import pytest

from src.config import GuardsSettings, settings
from src.risk.guards import RiskConfig, RiskGuards


def _set_guards(monkeypatch: pytest.MonkeyPatch, **overrides: object) -> None:
    base = {
        "max_orders_per_min": 30,
        "daily_loss_cap": 9_999_999.0,
        "trading_start_hhmm": "00:00",
        "trading_end_hhmm": "23:59",
        "kill_env": True,
        "kill_file": "",
    }
    base.update(overrides)
    monkeypatch.setattr(settings, "guards", GuardsSettings(**base))


def test_rate_limit_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_guards(monkeypatch, max_orders_per_min=2)
    guards = RiskGuards(RiskConfig())
    assert guards.ok_to_trade()
    assert guards.ok_to_trade()
    assert not guards.ok_to_trade()


def test_daily_loss_cap() -> None:
    guards = RiskGuards(
        RiskConfig(
            daily_loss_cap=100,
            trading_start_hm="00:00",
            trading_end_hm="23:59",
            kill_env=True,
            kill_file="",
        )
    )
    guards.set_pnl_today(-150)
    assert not guards.ok_to_trade()


def test_kill_switch_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_guards(monkeypatch, kill_env=False)
    guards = RiskGuards(RiskConfig())
    assert not guards.ok_to_trade()


def test_trading_window(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_guards(monkeypatch, trading_start_hhmm="10:00", trading_end_hhmm="10:01")
    guards = RiskGuards(RiskConfig())

    def fake_strftime(fmt: str, ts: float) -> str:  # pragma: no cover - simple stub
        return "09:00"

    monkeypatch.setattr(time, "strftime", fake_strftime)
    assert not guards.ok_to_trade()

    monkeypatch.setattr(time, "strftime", lambda fmt, ts: "10:00")
    assert guards.ok_to_trade()
