"""Unit coverage for the execution.bracket_manager module."""

from __future__ import annotations

import threading
import time
from unittest.mock import Mock

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager


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
