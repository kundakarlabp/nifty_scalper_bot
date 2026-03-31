"""Unit coverage for the execution.bracket_manager module."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


class TestBracketManager:
    """Behavioural checks for the BracketManager component."""

    def test_register_bracket_tracks_entry(self) -> None:
        """Bracket registration should store keyed state."""

        broker = Mock()
        manager = BracketManager(broker_client=broker)
        manager.register_bracket(
            entry_order_id="entry-1",
            stop_loss_order_id="sl-1",
            target_order_id="tp-1",
            entry_quantity=10,
        )

        assert "entry-1" in manager.brackets

    def test_stop_loss_fill_cancels_target(self) -> None:
        """Stop-loss fill should cancel pending target legs."""

        broker = Mock()
        manager = BracketManager(broker_client=broker)
        manager.register_bracket(
            entry_order_id="entry-2",
            stop_loss_order_id="sl-2",
            target_order_id="tp-2",
            entry_quantity=5,
        )

        manager.handle_bracket_update(
            order_id="sl-2",
            status="FILLED",
            filled_quantity=5,
        )

        broker.cancel_order.assert_called_once_with("tp-2")

    def test_create_bracket_normalizes_long_side(self) -> None:
        """create_bracket should normalize LONG to BUY with defaults."""
        broker = Mock()
        manager = BracketManager(broker_client=broker)

        order_id = manager.create_bracket(
            symbol="NIFTYTEST",
            side="LONG",
            entry_price=100.0,
            stop_loss=0.0,
            take_profit=0.0,
            quantity=1,
            strategy="unit_test",
        )

        bracket = manager.get_bracket(order_id)

        assert bracket is not None
        assert bracket.side == "BUY"
        assert bracket.sl_trigger_price == pytest.approx(95.0)
        assert bracket.tp_trigger_price == pytest.approx(110.0)

    @pytest.mark.flaky(reruns=1)
    def test_race_between_stop_and_target_is_serialised(self) -> None:
        """Only the first leg fill should trigger cancellations."""

        broker = Mock()
        manager = BracketManager(broker_client=broker)
        manager.register_bracket(
            entry_order_id="entry-3",
            stop_loss_order_id="sl-3",
            target_order_id="tp-3",
            entry_quantity=9,
        )

        def fill_stop() -> None:
            manager.handle_bracket_update(
                order_id="sl-3",
                status="FILLED",
                filled_quantity=9,
            )

        def fill_target() -> None:
            time.sleep(0.001)
            manager.handle_bracket_update(
                order_id="tp-3",
                status="FILLED",
                filled_quantity=9,
            )

        stop_thread = threading.Thread(target=fill_stop)
        target_thread = threading.Thread(target=fill_target)
        stop_thread.start()
        target_thread.start()
        stop_thread.join()
        target_thread.join()

        assert broker.cancel_order.call_count == 1

    def test_trailing_math_repairs_sell_lowest_ltp_inf(self) -> None:
        """Trailing math should repair invalid low-water marks for SELL."""

        order_manager = Mock()
        manager = BracketManager(order_manager=order_manager)
        bracket = BracketState(
            entry_order_id="entry-4",
            symbol="NIFTYTEST",
            side="SELL",
            quantity=75,
            entry_price=100.0,
            sl_trigger_price=110.0,
            tp_trigger_price=90.0,
            trailing_enabled=True,
            last_ltp=98.0,
            lowest_ltp=float("inf"),
        )

        manager._apply_trailing_math(bracket)

        assert bracket.lowest_ltp != float("inf")
        assert bracket.lowest_ltp == pytest.approx(98.0)
