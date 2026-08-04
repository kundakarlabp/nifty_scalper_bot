from __future__ import annotations

import logging
import types
from types import SimpleNamespace

from nifty_scalper_bot.execution.order_manager import (
    OrderDetails,
    OrderManager,
    OrderStatus,
    OrderType,
    TradePlan,
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
    assert plan.trade_provenance["tp1_status"] == "armed"


def test_single_lot_plan_does_not_create_fractional_tp1() -> None:
    plan = _plan(quantity=65)

    _enrich_trade_plan_exit_provenance(plan)

    assert "tp1_price" not in plan.trade_provenance
    assert "tp1_qty" not in plan.trade_provenance
    assert plan.trade_provenance["tp1_status"] == "skipped"
    assert plan.trade_provenance["tp1_skip_reason"] == "single_lot"


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



def test_fill_registration_passes_tp1_through_actual_order_manager_path() -> None:
    class CapturingBracketManager:
        def __init__(self) -> None:
            self.bracket = None
            self.registration = {}

        def get_bracket(self, order_id):
            return self.bracket

        def has_active_bracket(self, symbol):
            return False

        def register_virtual_bracket(self, **kwargs):
            self.registration = dict(kwargs)
            self.bracket = SimpleNamespace(entry_confirmed=False, active=False)

        def confirm_entry_fill(self, order_id, fill_price, filled_qty=None):
            self.bracket.entry_confirmed = True
            self.bracket.active = True
            return True

    bracket_manager = CapturingBracketManager()
    manager = SimpleNamespace(
        _bracket_manager=bracket_manager,
        _logger=logging.getLogger("tp1-fill-registration-test"),
        _notify_bracket_event=lambda *args, **kwargs: None,
    )
    order = OrderDetails(
        order_id="OID-TP1",
        symbol="NFO:NIFTY2680424400CE",
        side="BUY",
        quantity=130,
        order_type=OrderType.LIMIT,
        status=OrderStatus.FILLED,
        price=100.0,
        fill_price=100.0,
        average_price=100.0,
        filled_quantity=130,
        stop_loss=90.0,
        take_profit=120.0,
        intent="ENTRY",
        resolved_lot_size=65,
        trade_provenance={
            "tp1_price": 110.0,
            "tp1_qty": 65,
            "trailing_atr_mult": 1.25,
            "resolved_lot_size": 65,
            "tp1_status": "armed",
        },
    )

    OrderManager._register_virtual_bracket_for_fill(
        manager,
        order,
        source="test",
    )

    assert bracket_manager.registration["tp1_price"] == 110.0
    assert bracket_manager.registration["tp1_qty"] == 65
    assert bracket_manager.registration["trailing_atr_mult"] == 1.25
    assert bracket_manager.registration["resolved_lot_size"] == 65


def test_unresolved_lot_size_records_tp1_skip_reason() -> None:
    plan = _plan(quantity=130)
    plan.resolved_lot_size = 0

    _enrich_trade_plan_exit_provenance(plan)

    assert plan.trade_provenance["tp1_status"] == "skipped"
    assert plan.trade_provenance["tp1_skip_reason"] == "lot_size_unresolved"

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
