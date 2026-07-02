"""Tests for risk assessment helpers."""

from __future__ import annotations

from dataclasses import dataclass

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk import RiskManager
from nifty_scalper_bot.risk.assess_risk import assess_risk_soft_clear


@dataclass
class _DummyPositionManager:
    realized: float = 0.0

    def get_all_positions(self) -> list[object]:
        return []

    def get_realized_pnl(self) -> float:
        return self.realized

    def get_total_exposure(self) -> float:
        return 0.0


def _make_risk_manager(
    *,
    settings: RiskSettings,
    position_manager: _DummyPositionManager,
    balance: float,
) -> RiskManager:
    risk = RiskManager(
        settings=settings,
        position_manager=position_manager,
        account_balance=balance,
    )
    risk.set_lot_size_provider(lambda _symbol: 75, symbol="NIFTY")
    return risk


def test_reset_on_start_clears_soft(monkeypatch) -> None:
    risk = _make_risk_manager(
        settings=RiskSettings(per_trade_risk_pct=1.0),
        position_manager=_DummyPositionManager(),
        balance=100_000.0,
    )
    risk.record_rejection("COOLDOWN")
    assert risk.cooldown_remaining() == 0.0

    ok, detail, meta = assess_risk_soft_clear(risk)

    assert ok
    assert detail == "ok"
    assert meta["cooldown"] == 0.0
