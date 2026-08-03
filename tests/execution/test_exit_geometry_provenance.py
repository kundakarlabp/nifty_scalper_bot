from __future__ import annotations

import types

from nifty_scalper_bot.execution.order_manager import TradePlan
from nifty_scalper_bot.execution.premium_risk_contract_patch import (
    _enrich_virtual_bracket_kwargs,
    install_bracket_exit_provenance_hardening,
)
from nifty_scalper_bot.execution import runtime_order_manager as runtime_module
from nifty_scalper_bot.execution.runtime_order_manager import (
    _enrich_trade_plan_exit_provenance,
    _submit_core_with_exit_provenance,
)


def _plan(*, quantity: int = 130, provenance=None) -> TradePlan:
    return TradePlan(
        symbol="NFO:NIFTY2680424400CE",
        side="BUY",
        quantity=quantity,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        requested_lots=quantity // 65,
        resolved_lot_size=65,
        trade_provenance=dict(provenance or {}),
    )


def test_multilot_plan_gets_lot_aligned_tp1() -> None:
    plan = _plan(quantity=130)

    _enrich_trade_plan_exit_provenance(plan)

    assert plan.trade_provenance["resolved_lot_size"] == 65
    assert plan.trade_provenance["tp1_price"] == 110.0
    assert plan.trade_provenance["tp1_qty"] == 65
    assert plan.trade_provenance["initial_reward_risk"] == 2.0


def test_single_lot_plan_does_not_create_fractional_tp1() -> None:
    plan = _plan(quantity=65)

    _enrich_trade_plan_exit_provenance(plan)

    assert "tp1_price" not in plan.trade_provenance
    assert "tp1_qty" not in plan.trade_provenance


def test_existing_exit_provenance_is_preserved() -> None:
    plan = _plan(
        quantity=195,
        provenance={"tp1_price": 108.0, "tp1_qty": 65},
    )

    _enrich_trade_plan_exit_provenance(plan)

    assert plan.trade_provenance["tp1_price"] == 108.0
    assert plan.trade_provenance["tp1_qty"] == 65


def test_configured_atr_trailing_is_persisted(monkeypatch) -> None:
    monkeypatch.setenv("BRACKET_TRAILING_ATR_MULT", "1.25")
    plan = _plan(quantity=130)

    _enrich_trade_plan_exit_provenance(plan)

    assert plan.trade_provenance["trailing_atr_mult"] == 1.25


def test_fill_registration_restores_optional_exit_fields_from_provenance() -> None:
    kwargs = _enrich_virtual_bracket_kwargs(
        {
            "order_id": "OID-1",
            "symbol": "NFO:NIFTY2680424400CE",
            "side": "BUY",
            "qty": 130,
            "price": 100.0,
            "sl": 90.0,
            "tp": 120.0,
            "trade_provenance": {
                "tp1_price": 110.0,
                "tp1_qty": 65,
                "trailing_atr_mult": 1.25,
                "resolved_lot_size": 65,
            },
        }
    )

    assert kwargs["tp1_price"] == 110.0
    assert kwargs["tp1_qty"] == 65
    assert kwargs["trailing_atr_mult"] == 1.25
    assert kwargs["resolved_lot_size"] == 65


def test_explicit_registration_values_override_provenance() -> None:
    kwargs = _enrich_virtual_bracket_kwargs(
        {
            "tp1_price": 109.0,
            "tp1_qty": 65,
            "resolved_lot_size": 65,
            "trade_provenance": {
                "tp1_price": 110.0,
                "tp1_qty": 130,
                "resolved_lot_size": 75,
            },
        }
    )

    assert kwargs["tp1_price"] == 109.0
    assert kwargs["tp1_qty"] == 65
    assert kwargs["resolved_lot_size"] == 65


def test_bracket_installer_preserves_existing_composed_registration() -> None:
    calls = []

    class Bracket:
        def register_virtual_bracket(self, *args, **kwargs):
            calls.append((args, kwargs))
            return "registered"

    original = Bracket.register_virtual_bracket
    install_bracket_exit_provenance_hardening(Bracket)
    installed = Bracket.register_virtual_bracket
    install_bracket_exit_provenance_hardening(Bracket)

    result = Bracket().register_virtual_bracket(
        order_id="OID-2",
        trade_provenance={
            "tp1_price": 110.0,
            "tp1_qty": 65,
            "resolved_lot_size": 65,
        },
    )

    assert result == "registered"
    assert installed is Bracket.register_virtual_bracket
    assert installed is not original
    assert len(calls) == 1
    assert calls[0][1]["tp1_price"] == 110.0
    assert calls[0][1]["tp1_qty"] == 65
    assert calls[0][1]["resolved_lot_size"] == 65


def test_recovery_submit_enriches_rebuilt_plan(monkeypatch) -> None:
    captured = {}

    def fake_submit(manager, plan):
        captured["manager"] = manager
        captured["plan"] = plan
        return types.SimpleNamespace(accepted=True, order_id="OID-3")

    monkeypatch.setattr(
        runtime_module._core.OrderManager,
        "submit_trade_plan_result",
        fake_submit,
    )
    manager = object()
    rebuilt = _plan(quantity=130)

    result = _submit_core_with_exit_provenance(manager, rebuilt)

    assert result.order_id == "OID-3"
    assert captured["manager"] is manager
    assert captured["plan"] is rebuilt
    assert rebuilt.trade_provenance["tp1_price"] == 110.0
    assert rebuilt.trade_provenance["tp1_qty"] == 65
