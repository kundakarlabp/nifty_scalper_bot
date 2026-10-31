"""Canonical OrderManager and BracketManager linked screens never submit."""

from types import SimpleNamespace

from nifty_scalper_bot.execution.bracket_core import BracketManager
from nifty_scalper_bot.execution.margin_engine import LinkedPortfolioMarginDecision
from nifty_scalper_bot.execution.order_manager_core import OrderManager


class Positions:
    def __init__(self):
        self.parent = {
            "state": "RESERVED_UNVERIFIED",
            "legs": {
                "long_call": {"order_id": None},
                "long_put": {"order_id": None},
            },
        }
        self.blocker = None

    def get_linked_portfolio_reservation(self, parent_id):
        return self.parent

    def current_pnl_reconciliation_blocker(self):
        return self.blocker

    def linked_portfolio_recovery_plan(self, parent_id):
        return {
            "state": "PARTIAL_EXPOSURE_RECOVERY_REQUIRED",
            "reason": "linked_child_not_fully_filled",
            "group_exit_intents": [
                {"role": "short_call", "side": "BUY", "quantity": 25},
                {"role": "long_call", "side": "SELL", "quantity": 65},
            ],
            "cancel_order_ids": [],
        }


class Risk:
    def __init__(self):
        self.green = True
        self.snapshot_value = SimpleNamespace(
            shadow_forced=False,
            breaker_tripped=False,
            per_trade_risk_pct=2.0,
            daily_loss_limit=10000.0,
            day_loss=1000.0,
            daily_realized=-500.0,
        )
        self.account_balance = 1000000.0

    def is_green(self):
        return self.green

    def risk_gate_should_trade(self):
        return True, ()

    def snapshot(self):
        return self.snapshot_value


class Margin:
    def __init__(self):
        self.calls = []

    def screen_linked_portfolio(self, **kwargs):
        self.calls.append(kwargs)
        return LinkedPortfolioMarginDecision(
            ok=True,
            reason="linked_margin_authority_observed",
            initial_margin=120000,
            final_margin=58000,
            available_cash=140000,
            required_cash_with_reserves=120700,
            stressed_loss_with_costs=13700,
            observed_at_utc="2026-10-31T09:00:00+00:00",
        )


def owner():
    positions = Positions()
    risk = Risk()
    margin = Margin()
    manager = SimpleNamespace(
        _positions=positions, _risk_manager=risk, _margin_engine=margin
    )
    return manager, positions, risk, margin


def screen(manager):
    return OrderManager.screen_linked_portfolio_entry(
        manager,
        "parent-001",
        product="NRML",
        gross_loss_bound=13000,
        entry_fee_reserve=100,
        exit_fee_reserve=200,
        stressed_slippage_reserve=400,
    )


def test_order_manager_uses_risk_owner_and_does_not_place():
    manager, positions, risk, margin = owner()
    view = screen(manager)
    assert view["risk_screen_passed"] is True
    assert view["broker_execution_allowed"] is False
    assert view["live_entry_allowed"] is False
    assert margin.calls[0]["remaining_trade_risk"] == 20000
    assert margin.calls[0]["remaining_daily_risk"] == 9000
    assert margin.calls[0]["legs"] is positions.parent["legs"]


def test_parent_already_started_and_pnl_uncertainty_are_fail_closed():
    manager, positions, risk, margin = owner()
    positions.parent["state"] = "IN_PROGRESS_UNVERIFIED"
    assert screen(manager)["reason"] == (
        "linked_parent_in_progress_requires_reconciliation"
    )
    assert margin.calls == []
    positions.parent["state"] = "RESERVED_UNVERIFIED"
    positions.blocker = "broker_unverified"
    assert screen(manager)["reason"] == "linked_pnl_authority_unverified"


def test_live_risk_manager_block_cannot_be_ignored():
    manager, positions, risk, margin = owner()
    risk.green = False
    assert screen(manager)["reason"] == "linked_risk_gate_closed"
    assert margin.calls == []
    risk.green = True
    risk.snapshot_value.daily_loss_limit = 500
    assert screen(manager)["reason"] == "linked_risk_budget_exhausted"
    assert margin.calls == []


def test_bracket_group_diagnostic_preserves_short_first_and_no_dispatch():
    positions = Positions()
    bracket = SimpleNamespace(order_manager=SimpleNamespace(_positions=positions))
    result = BracketManager.linked_portfolio_exit_convergence(
        bracket, "parent-001"
    )
    assert result["exit_converged"] is False
    assert result["live_exit_dispatch_allowed"] is False
    assert [x["role"] for x in result["exit_intents"]] == [
        "short_call",
        "long_call",
    ]


def test_bracket_group_diagnostic_refuses_exit_on_ambiguous_order():
    positions = Positions()

    def ambiguous(parent_id):
        return {
            "state": "CANCEL_THEN_RECONCILE",
            "reason": "inflight_child_may_still_fill",
            "cancel_order_ids": ["child-1"],
            "group_exit_intents": [{"role": "long_call", "quantity": 65}],
        }

    positions.linked_portfolio_recovery_plan = ambiguous
    bracket = SimpleNamespace(order_manager=SimpleNamespace(_positions=positions))
    result = BracketManager.linked_portfolio_exit_convergence(
        bracket, "parent-001"
    )
    assert result["exit_intents"] == []
    assert result["cancel_order_ids"] == ["child-1"]
    assert result["live_exit_dispatch_allowed"] is False
