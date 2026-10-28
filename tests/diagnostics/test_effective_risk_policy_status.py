"""Read-only risk provenance and balance-bound exposure diagnostics."""

from types import SimpleNamespace

from nifty_scalper_bot.main import _effective_risk_policy_status


def _context(pct: float, *, balance_valid: bool = True):
    manager = SimpleNamespace(
        settings=SimpleNamespace(per_trade_risk_pct=pct),
        _switches=SimpleNamespace(max_day_loss=808.17),
    )
    return SimpleNamespace(
        risk_manager=manager,
        broker_balance_valid=balance_valid,
        last_valid_broker_balance=16163.45,
    )


def test_effective_risk_reports_canonical_five_percent_without_reconfiguring() -> None:
    ctx = _context(5.0)
    before = ctx.risk_manager.settings.per_trade_risk_pct
    result = _effective_risk_policy_status(ctx)
    assert result["known"] is True
    assert result["per_trade_risk_pct"] == 5.0
    assert result["per_trade_budget_rupees"] == 808.17
    assert result["max_day_loss_rupees"] == 808.17
    assert result["source"] == "github_policy_or_equal_override"
    assert result["authority"] == "RiskManager"
    assert ctx.risk_manager.settings.per_trade_risk_pct == before


def test_stricter_effective_config_remains_visible_without_widening_risk() -> None:
    ctx = _context(1.0)
    result = _effective_risk_policy_status(ctx)
    assert result["source"] == "stricter_effective_setting"
    assert result["per_trade_budget_rupees"] == 161.63
    assert ctx.risk_manager.settings.per_trade_risk_pct == 1.0


def test_missing_broker_verification_never_invents_rupee_budget() -> None:
    result = _effective_risk_policy_status(_context(5.0, balance_valid=False))
    assert result["per_trade_budget_rupees"] is None
    assert result["validated_balance"] is None


def test_missing_risk_manager_is_explicitly_unknown() -> None:
    result = _effective_risk_policy_status(SimpleNamespace())
    assert result == {"known": False, "source": "risk_manager_unavailable"}
