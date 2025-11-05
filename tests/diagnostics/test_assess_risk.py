"""Tests for risk assessment helpers."""

from __future__ import annotations

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk.assess_risk import assess_risk_soft_clear
from tests.test_risk_manager_new import DummyPositionManager, _make_risk_manager


def test_reset_on_start_clears_soft(monkeypatch):
    risk = _make_risk_manager(
        settings=RiskSettings(per_trade_risk_pct=1.0),
        pm=DummyPositionManager(realized=0.0),
        balance=100_000.0,
    )
    risk.record_rejection("COOLDOWN")
    assert risk.cooldown_remaining() == 0.0
    ok, detail, meta = assess_risk_soft_clear(risk)
    assert ok
    assert detail == "ok"
    assert meta["cooldown"] == 0.0
