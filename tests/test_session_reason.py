from __future__ import annotations

from datetime import datetime, timezone

from nifty_scalper_bot.core.app import TradingSessionStatus, _resolve_session_reason
from nifty_scalper_bot.risk import RiskSnapshot


def _make_status(**overrides: object) -> TradingSessionStatus:
    base = dict(
        session_valid=True,
        rate_limits_ok=True,
        market_open=True,
        risk_green=False,
        reasons=["COOLDOWN"],
        timestamp=datetime.now(timezone.utc),
        override_out_of_hours=False,
        fail_reason="COOLDOWN",
    )
    base.update(overrides)
    return TradingSessionStatus(**base)  # type: ignore[arg-type]


def _make_snapshot(**overrides: object) -> RiskSnapshot:
    base = dict(
        daily_realized=0.0,
        daily_loss_limit=0.0,
        day_loss=0.0,
        max_day_loss=0.0,
        losses_in_row=0,
        cooldown_remaining=5.0,
        breaker_tripped=False,
        breaker_reason=None,
        shadow_forced=False,
        per_trade_risk_pct=0.0,
        last_rejection="COOLDOWN",
        timestamp=datetime.now(timezone.utc),
    )
    base.update(overrides)
    return RiskSnapshot(**base)  # type: ignore[arg-type]


def test_resolve_reason_soft_override() -> None:
    status = _make_status()
    snapshot = _make_snapshot()

    reason, soft_override = _resolve_session_reason(status, snapshot)

    assert reason == "COOLDOWN"
    assert soft_override is True


def test_resolve_reason_breaker_is_hard() -> None:
    status = _make_status(reasons=["BREAKER"], fail_reason="BREAKER")
    snapshot = _make_snapshot(
        breaker_tripped=True,
        breaker_reason="Daily loss limit reached",
        cooldown_remaining=0.0,
    )

    reason, soft_override = _resolve_session_reason(status, snapshot)

    assert reason == "BREAKER"
    assert soft_override is False


def test_resolve_reason_without_snapshot() -> None:
    status = _make_status(
        reasons=["RISK_CHECK_FAILED"],
        fail_reason="RISK_CHECK_FAILED",
    )

    reason, soft_override = _resolve_session_reason(status, None)

    assert reason == "RISK_CHECK_FAILED"
    assert soft_override is False


def test_session_status_as_dict_reflects_invalid_session() -> None:
    """Ensure serialization exposes invalid sessions.

    Args:
        None
    Returns:
        None
    Raises:
        AssertionError: If serialization flags are unexpected.
    """

    status = _make_status(
        session_valid=False,
        market_open=True,
        override_out_of_hours=False,
    )

    serialized = status.as_dict()

    assert serialized["session_valid"] is False
    assert serialized["broker_session_valid"] is False
    assert serialized["market_open"] is True
