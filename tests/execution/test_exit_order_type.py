from __future__ import annotations

import logging
from unittest.mock import MagicMock

from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


def test_exit_order_type_uses_market_for_sl() -> None:
    """Verify SL order type. Args: None. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        order_manager = MagicMock()
        manager = BracketManager(order_manager=order_manager)

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

        manager._execute_exit_order(
            bracket=bracket,
            qty=10,
            exit_side="SELL",
            reason="SL Hit",
            is_partial=False,
            exit_type="SL",
        )

        kwargs = order_manager.place_order.call_args.kwargs
        assert kwargs["order_type"] == "MARKET"
        assert "price" not in kwargs
    except Exception as e:
        logger.error("Failure in test_exit_order_type_uses_market_for_sl: %s", e)
        raise


def test_exit_order_type_uses_limit_for_tp() -> None:
    """Verify TP order type. Args: None. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        order_manager = MagicMock()
        manager = BracketManager(order_manager=order_manager)

        bracket = BracketState(
            entry_order_id="entry-2",
            symbol="NFO:TEST",
            side="BUY",
            quantity=10,
            entry_price=100.0,
            sl_trigger_price=95.0,
            tp_trigger_price=110.0,
            remaining_quantity=10,
        )
        bracket.last_ltp = 108.0

        manager._execute_exit_order(
            bracket=bracket,
            qty=10,
            exit_side="SELL",
            reason="TP Hit",
            is_partial=False,
            exit_type="FINAL_TP",
        )

        kwargs = order_manager.place_order.call_args.kwargs
        assert kwargs["order_type"] == "LIMIT"
        assert kwargs["price"] > 0
    except Exception as e:
        logger.error("Failure in test_exit_order_type_uses_limit_for_tp: %s", e)
        raise
