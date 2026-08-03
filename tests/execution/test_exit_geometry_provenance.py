from __future__ import annotations

from nifty_scalper_bot.execution.bracket_manager import (
    _enrich_virtual_bracket_kwargs,
)
from nifty_scalper_bot.execution.order_manager import TradePlan
from nifty_scalper_bot.execution.runtime_order_manager import (
    _enrich_trade_plan_exit_provenance,
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
