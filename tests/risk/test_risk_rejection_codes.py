"""Tests covering risk rejection cooldown handling and canonical codes."""

from __future__ import annotations

from datetime import datetime, timezone
import time

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


def test_refresh_account_balance_returns_cached_value_within_ttl() -> None:
    class _Hub:
        def __init__(self) -> None:
            self.calls = 0

        def get_available_balance(self, *, force: bool = False) -> float:
            self.calls += 1
            return 75_000.0

    risk = _make_risk_manager()
    hub = _Hub()
    risk._data_hub = hub
    risk._cached_balance = 50_000.0
    risk.account_balance = 50_000.0
    risk._last_balance_refresh = time.time()
    risk._balance_cache_ttl = 60.0

    assert risk.refresh_account_balance() == 50_000.0
    assert hub.calls == 0
