"""Regression tests for post-fill virtual-bracket risk geometry."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

from nifty_scalper_bot.execution.bracket_manager import BracketManager
from nifty_scalper_bot.execution.runtime_order_manager import (
    _enrich_trade_plan_exit_provenance,
)


def _manager() -> BracketManager:
    order_manager = Mock()
    order_manager.is_live_mode.return_value = False
    return BracketManager(order_manager=order_manager)


def test_buy_fill_preserves_absolute_stop_and_target_distances() -> None:
    manager = _manager()
    manager.register_virtual_bracket(
        order_id="buy-fill",
        symbol="NFO:NIFTYTESTCE",
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        trade_provenance={"bracket_anchor_mode": "distance_from_entry"},
    )

    manager.confirm_entry_fill("buy-fill", 110.0)

    bracket = manager.get_bracket("buy-fill")
    assert bracket is not None
    assert bracket.entry_price == 110.0
    assert bracket.sl_trigger_price == 100.0
    assert bracket.initial_sl_trigger_price == 100.0
    assert bracket.tp_trigger_price == 130.0


def test_sell_fill_preserves_absolute_stop_and_target_distances() -> None:
    manager = _manager()
    manager.register_virtual_bracket(
        order_id="sell-fill",
        symbol="NFO:NIFTYTESTPE",
        side="SELL",
        qty=65,
        price=100.0,
        sl=110.0,
        tp=80.0,
        trade_provenance={"bracket_anchor_mode": "distance_from_entry"},
    )

    manager.confirm_entry_fill("sell-fill", 90.0)

    bracket = manager.get_bracket("sell-fill")
    assert bracket is not None
    assert bracket.entry_price == 90.0
    assert bracket.sl_trigger_price == 100.0
    assert bracket.initial_sl_trigger_price == 100.0
    assert bracket.tp_trigger_price == 70.0


def test_absolute_level_bracket_does_not_move_levels_on_fill() -> None:
    manager = _manager()
    manager.register_virtual_bracket(
        order_id="absolute-fill",
        symbol="NFO:NIFTYABSCE",
        side="BUY",
        qty=65,
        price=100.0,
        sl=90.0,
        tp=120.0,
        trade_provenance={"bracket_anchor_mode": "absolute_level"},
    )

    manager.confirm_entry_fill("absolute-fill", 110.0)

    bracket = manager.get_bracket("absolute-fill")
    assert bracket is not None
    assert bracket.entry_price == 110.0
    assert bracket.sl_trigger_price == 90.0
    assert bracket.initial_sl_trigger_price == 90.0
    assert bracket.tp_trigger_price == 120.0


def test_runtime_provenance_carries_bracket_anchor_mode() -> None:
    plan = SimpleNamespace(
        intent="ENTRY",
        trade_provenance={},
        quantity=65,
        resolved_lot_size=65,
        entry_price=100.0,
        stop_loss=90.0,
        take_profit=120.0,
        side="BUY",
        bracket_anchor_mode="absolute_level",
    )

    _enrich_trade_plan_exit_provenance(plan)

    assert plan.trade_provenance["bracket_anchor_mode"] == "absolute_level"
