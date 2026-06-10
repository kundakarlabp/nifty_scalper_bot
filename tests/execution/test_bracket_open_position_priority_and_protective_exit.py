from __future__ import annotations

from unittest.mock import MagicMock

from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


def _active_bracket(manager: BracketManager, *, side: str = "BUY") -> BracketState:
    bracket = BracketState(
        entry_order_id="entry-1",
        symbol="NFO:TEST",
        side=side,
        quantity=10,
        entry_price=100.0,
        sl_trigger_price=95.0,
        tp_trigger_price=110.0,
        remaining_quantity=10,
        active=False,
        entry_confirmed=False,
    )
    manager._brackets[bracket.entry_order_id] = bracket
    return bracket


def test_confirm_entry_fill_registers_open_position_priority_once() -> None:
    manager = BracketManager(order_manager=MagicMock())
    manager.shutdown()
    bracket = _active_bracket(manager)
    opened: list[str] = []
    manager.attach_open_position_priority_hooks(on_position_open=opened.append)

    manager.confirm_entry_fill(bracket.entry_order_id, 101.0)

    assert opened == ["NFO:TEST"]
    assert bracket.active is True


def test_bracket_close_removes_open_position_priority() -> None:
    manager = BracketManager(order_manager=MagicMock())
    manager.shutdown()
    bracket = _active_bracket(manager)
    closed: list[str] = []
    manager.attach_open_position_priority_hooks(on_position_closed=closed.append)

    manager._close_bracket(bracket, close_source="unit_test")

    assert closed == ["NFO:TEST"]
    assert bracket.active is False


def test_priority_hook_exception_does_not_block_activation() -> None:
    manager = BracketManager(order_manager=MagicMock())
    manager.shutdown()
    bracket = _active_bracket(manager)
    manager.attach_open_position_priority_hooks(on_position_open=lambda _symbol: (_ for _ in ()).throw(RuntimeError("boom")))

    manager.confirm_entry_fill(bracket.entry_order_id, 101.0)

    assert bracket.active is True


def test_hard_sl_protective_exit_defaults_to_market(monkeypatch) -> None:
    monkeypatch.delenv("EXIT_PROTECTIVE_ORDER_MODE", raising=False)
    order_manager = MagicMock()
    order_manager.place_order.return_value = "exit-1"
    manager = BracketManager(order_manager=order_manager)
    manager.shutdown()
    _active_bracket(manager)

    result = manager.submit_exit_order("NFO:TEST", 10, "HARD_SL_BREACH", "entry-1", preferred_order_type="LIMIT")

    assert result.accepted is True
    assert order_manager.place_order.call_args.kwargs["order_type"] == "MARKET"
    assert "price" not in order_manager.place_order.call_args.kwargs


def test_eod_flatten_protective_exit_defaults_to_market(monkeypatch) -> None:
    monkeypatch.delenv("EXIT_PROTECTIVE_ORDER_MODE", raising=False)
    order_manager = MagicMock()
    order_manager.place_order.return_value = "exit-1"
    manager = BracketManager(order_manager=order_manager)
    manager.shutdown()
    _active_bracket(manager)

    manager.submit_exit_order("NFO:TEST", 10, "EOD_FLATTEN", "entry-1", preferred_order_type="LIMIT")

    assert order_manager.place_order.call_args.kwargs["order_type"] == "MARKET"


def test_aggressive_limit_sell_exit_prices_below_bid(monkeypatch) -> None:
    monkeypatch.setenv("EXIT_PROTECTIVE_ORDER_MODE", "AGGRESSIVE_LIMIT")
    order_manager = MagicMock()
    order_manager.place_order.return_value = "exit-1"
    market_data = MagicMock()
    market_data.get_quote.return_value = {"bid": 100.0, "ask": 101.0, "ltp": 100.5}
    manager = BracketManager(order_manager=order_manager, market_data=market_data)
    manager.shutdown()
    _active_bracket(manager, side="BUY")

    manager.submit_exit_order("NFO:TEST", 10, "WATCHDOG_HARD_SL", "entry-1", preferred_order_type="LIMIT")

    kwargs = order_manager.place_order.call_args.kwargs
    assert kwargs["order_type"] == "LIMIT"
    assert kwargs["price"] < 100.0


def test_aggressive_limit_quote_missing_falls_back_to_market(monkeypatch) -> None:
    monkeypatch.setenv("EXIT_PROTECTIVE_ORDER_MODE", "AGGRESSIVE_LIMIT")
    monkeypatch.setenv("EXIT_FALLBACK_TO_MARKET_ON_QUOTE_MISSING", "true")
    order_manager = MagicMock()
    order_manager.place_order.return_value = "exit-1"
    market_data = MagicMock()
    market_data.get_quote.return_value = {}
    manager = BracketManager(order_manager=order_manager, market_data=market_data)
    manager.shutdown()
    _active_bracket(manager)

    manager.submit_exit_order("NFO:TEST", 10, "FORCED_SL_EXIT", "entry-1", preferred_order_type="LIMIT")

    assert order_manager.place_order.call_args.kwargs["order_type"] == "MARKET"


def test_legacy_execute_exit_order_removed_from_runtime_path() -> None:
    assert not hasattr(BracketManager, "_execute_exit_order")
