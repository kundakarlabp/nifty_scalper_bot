from __future__ import annotations

from datetime import datetime, time, timezone

import pytest

from nifty_scalper_bot.core.app import (
    TradingSessionGuard,
    _resolve_session_reason,
)
from nifty_scalper_bot.risk.risk_manager import RiskSnapshot
from nifty_scalper_bot.risk.session_gate import IST, build_session_guard, can_trade
from nifty_scalper_bot.utils.rate_limiter import RateLimiter


class _DummyRisk:
    def __init__(self, *, healthy: bool = True) -> None:
        self._healthy = healthy

    def is_green(self) -> bool:
        return self._healthy


def test_session_guard_trading_window_update() -> None:
    guard = TradingSessionGuard(rate_limiter=RateLimiter(), risk_manager=_DummyRisk())
    guard.set_trading_window("10:15", "14:30")
    open_time, close_time = guard.get_trading_window()
    assert open_time == time(10, 15)
    assert close_time == time(14, 30)


def test_session_guard_override_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = TradingSessionGuard(rate_limiter=RateLimiter(), risk_manager=_DummyRisk())
    guard.mark_session_valid()

    def _fake_guard(*_args, **kwargs):
        override = bool(kwargs.get("override", False))
        market_open = False
        session_valid = bool(market_open or override)
        reasons = [] if session_valid else ["Outside trading window"]
        return {
            "market_open": market_open,
            "override_out_of_hours": override,
            "session_valid": session_valid,
            "rate_limits_ok": True,
            "risk_green": True,
            "reasons": reasons,
            "timestamp": "2024-01-01T00:00:00+05:30",
        }

    monkeypatch.setattr("nifty_scalper_bot.core.app.build_session_guard", _fake_guard)

    status = guard.evaluate()
    assert status.override_out_of_hours is False
    assert any("Outside trading window" in reason for reason in status.reasons)

    guard.set_allow_out_of_hours(True)
    status = guard.evaluate()
    assert status.override_out_of_hours is True
    assert not status.reasons


def test_build_session_guard_market_hours() -> None:
    now = datetime(2024, 1, 1, 10, 0, tzinfo=IST)
    guard = build_session_guard(now=now)
    assert guard["market_open"] is True
    assert guard["session_valid"] is True
    assert guard["reasons"] == []


def test_build_session_guard_outside_hours(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SESSION_ALLOW_OUT_OF_HOURS", raising=False)
    monkeypatch.delenv("ALLOW_OFFHOURS_TESTING", raising=False)
    now = datetime(2024, 1, 1, 8, 0, tzinfo=IST)
    guard = build_session_guard(now=now)
    assert guard["market_open"] is False
    assert guard["session_valid"] is False
    assert guard["reasons"] == ["Outside trading window"]


def test_build_session_guard_respects_override_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SESSION_ALLOW_OUT_OF_HOURS", "true")
    now = datetime(2024, 1, 1, 8, 0, tzinfo=IST)
    guard = build_session_guard(now=now)
    assert guard["market_open"] is False
    assert guard["override_out_of_hours"] is True
    assert guard["session_valid"] is True
    assert guard["reasons"] == []


def test_can_trade_requires_all_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    guard = {
        "session_valid": True,
        "rate_limits_ok": True,
        "risk_green": True,
        "market_open": True,
        "broker_session_valid": True,
    }
    assert can_trade(guard, enable_live=True, shadow_mode=False) is True

    guard["rate_limits_ok"] = False
    assert can_trade(guard, enable_live=True, shadow_mode=False) is False

    guard["rate_limits_ok"] = True
    guard["risk_green"] = False
    assert can_trade(guard, enable_live=True, shadow_mode=False) is False

    guard["risk_green"] = True
    assert can_trade(guard, enable_live=False, shadow_mode=False) is False
    assert can_trade(guard, enable_live=True, shadow_mode=True) is False

    guard["broker_session_valid"] = False
    assert can_trade(guard, enable_live=True, shadow_mode=False) is False


def test_can_trade_defaults_to_false() -> None:
    guard: dict[str, object] = {}
    assert can_trade(guard, enable_live=True, shadow_mode=False) is False


def test_session_status_maps_soft_vs_hard(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Risk:
        def __init__(self) -> None:
            self.cooldown = 0.0
            self.breaker = False

        def is_green(self) -> bool:
            return not self.breaker and self.cooldown <= 0.0

        def snapshot(self):  # noqa: D401 - simple namespace
            from types import SimpleNamespace

            now = datetime.now(timezone.utc)
            return SimpleNamespace(
                daily_realized=0.0,
                daily_loss_limit=0.0,
                day_loss=0.0,
                max_day_loss=0.0,
                losses_in_row=0,
                cooldown_remaining=self.cooldown,
                breaker_tripped=self.breaker,
                breaker_reason="BREAKER" if self.breaker else None,
                shadow_forced=False,
                per_trade_risk_pct=0.0,
                last_rejection="STALE" if self.cooldown > 0 else None,
                timestamp=now,
            )

    def _always_open(*_args, **_kwargs) -> dict[str, object]:
        now = datetime.now(timezone.utc)
        return {
            "market_open": True,
            "override_out_of_hours": False,
            "session_valid": True,
            "rate_limits_ok": True,
            "risk_green": True,
            "reasons": [],
            "timestamp": now.isoformat(),
        }

    monkeypatch.setattr("nifty_scalper_bot.core.app.build_session_guard", _always_open)
    risk = _Risk()
    limiter = RateLimiter()
    monkeypatch.setattr(
        limiter,
        "snapshot",
        lambda: {"orders": {"tokens": 10.0, "capacity": 10.0}},
    )
    guard = TradingSessionGuard(rate_limiter=limiter, risk_manager=risk)
    guard.mark_session_valid()

    risk.cooldown = 2.0
    payload = guard.snapshot()
    assert payload["session_fail_reason"] == "COOLDOWN"

    risk.cooldown = 0.0
    risk.breaker = True
    payload_breaker = guard.snapshot()
    assert payload_breaker["session_fail_reason"] == "BREAKER"


def test_session_guard_allows_soft_override_in_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    now = datetime.now(timezone.utc)

    def _always_open(*_args, **_kwargs) -> dict[str, object]:
        return {
            "market_open": True,
            "override_out_of_hours": False,
            "session_valid": True,
            "rate_limits_ok": True,
            "risk_green": False,
            "reasons": ["COOLDOWN"],
            "timestamp": now.isoformat(),
        }

    monkeypatch.setattr("nifty_scalper_bot.core.app.build_session_guard", _always_open)

    class _SoftRisk:
        def __init__(self) -> None:
            self.snapshot_called = 0

        def is_green(self) -> bool:
            return False

        def snapshot(self) -> RiskSnapshot:
            self.snapshot_called += 1
            return RiskSnapshot(
                daily_realized=0.0,
                daily_loss_limit=0.0,
                day_loss=0.0,
                max_day_loss=0.0,
                losses_in_row=0,
                cooldown_remaining=9.5,
                breaker_tripped=False,
                breaker_reason=None,
                shadow_forced=False,
                per_trade_risk_pct=1.0,
                last_rejection="COOLDOWN",
                timestamp=now,
            )

    risk = _SoftRisk()
    limiter = RateLimiter()
    monkeypatch.setattr(
        limiter,
        "snapshot",
        lambda: {"orders": {"tokens": 10.0, "capacity": 10.0}},
    )

    guard = TradingSessionGuard(rate_limiter=limiter, risk_manager=risk)
    guard.mark_session_valid()

    allowed, status = guard.allow_live()

    assert allowed is False
    assert status.market_open is True

    reason, soft_override = _resolve_session_reason(status, risk.snapshot())

    assert reason == "COOLDOWN"
    assert soft_override is True
