from __future__ import annotations

from unittest.mock import MagicMock

from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


def _manager_with_bracket(monkeypatch, *, reason_mode: str | None = None) -> tuple[BracketManager, MagicMock]:
    if reason_mode is None:
        monkeypatch.delenv("EXIT_PROTECTIVE_ORDER_MODE", raising=False)
    else:
        monkeypatch.setenv("EXIT_PROTECTIVE_ORDER_MODE", reason_mode)
    order_manager = MagicMock()
    order_manager.place_order.return_value = "exit-1"
    manager = BracketManager(order_manager=order_manager)
    manager.shutdown()
    bracket = BracketState(
        entry_order_id="entry-1",
        symbol="NFO:TEST",
        side="BUY",
        quantity=10,
        entry_price=100.0,
        sl_trigger_price=95.0,
        tp_trigger_price=110.0,
        remaining_quantity=10,
    )
    bracket.last_ltp = 98.0
    manager._brackets[bracket.entry_order_id] = bracket
    return manager, order_manager


def test_exit_order_type_uses_market_for_sl(monkeypatch) -> None:
    manager, order_manager = _manager_with_bracket(monkeypatch)

    manager.submit_exit_order(
        symbol="NFO:TEST",
        qty=10,
        reason="HARD_SL_BREACH",
        bracket_id="entry-1",
        preferred_order_type="LIMIT",
    )

    kwargs = order_manager.place_order.call_args.kwargs
    assert kwargs["order_type"] == "MARKET"
    assert "price" not in kwargs


def test_exit_order_type_keeps_limit_for_take_profit(monkeypatch) -> None:
    manager, order_manager = _manager_with_bracket(monkeypatch)

    manager.submit_exit_order(
        symbol="NFO:TEST",
        qty=10,
        reason="HARD_TP_BREACH",
        bracket_id="entry-1",
        preferred_order_type="LIMIT",
    )

    kwargs = order_manager.place_order.call_args.kwargs
    assert kwargs["order_type"] == "LIMIT"
    assert kwargs["price"] > 0
