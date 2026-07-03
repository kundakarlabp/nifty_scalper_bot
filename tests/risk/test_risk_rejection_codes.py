"""Tests covering risk rejection cooldown handling and canonical codes."""

from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.config.settings import RiskSettings
from nifty_scalper_bot.risk.risk_manager import RiskManager


class DummyPositionManager:
    """Minimal position manager stub for risk manager tests."""

    def __init__(self, realized: float = 0.0) -> None:
        self.realized = realized

    def get_all_positions(self) -> list[object]:  # pragma: no cover - test stub
        return []

    def get_realized_pnl(self) -> float:
        return self.realized

    def get_total_exposure(self) -> float:
        return 0.0


def _make_risk_manager(cooldown_seconds: float = 5.0) -> RiskManager:
    settings = RiskSettings(
        per_trade_risk_pct=1.0,
        cooldown_on_reject_seconds=cooldown_seconds,
    )
    manager = DummyPositionManager(realized=0.0)
    return RiskManager(
        settings=settings,
        position_manager=manager,
        account_balance=50_000.0,
    )


def test_record_rejection_trading_disabled_is_soft() -> None:
    risk = _make_risk_manager()

    risk.record_rejection("Trading disabled by risk state")

    assert risk.cooldown_remaining() == 0.0
    assert risk._cooldown_until is None
    assert risk._switches.cooldown_remaining() == 0.0
    assert risk._last_rejection == "RISK_STATE"


def test_record_rejection_cooldown_only_on_broker_error() -> None:
    risk = _make_risk_manager(cooldown_seconds=30.0)
    before = datetime.now(timezone.utc)

    risk.record_rejection("BROKER_ERROR_5XX")

    assert risk.cooldown_remaining() > 0.0
    assert risk._cooldown_until is not None
    assert risk._cooldown_until > before
    assert risk._last_rejection == "BROKER_ERROR_5XX"


def test_record_rejection_margin_is_soft() -> None:
    risk = _make_risk_manager()

    risk.record_rejection("Insufficient funds available")

    assert risk.cooldown_remaining() == 0.0
    assert risk._cooldown_until is None
    assert risk._last_rejection == "MARGIN"


def test_risk_config_max_concurrent_positions_matches_enforced_single_position() -> None:
    """Slice-4: the second (dead-code) RiskManager's config default must not
    silently disagree with the single-position policy enforced at the
    execution choke point (order_manager_core's single-position gate)."""
    from nifty_scalper_bot.config.base import RiskConfig
    from nifty_scalper_bot.config import defaults

    assert RiskConfig().max_concurrent_positions == 1
    assert defaults.DEFAULT_RISK_MAX_CONCURRENT_POSITIONS == 1
