"""Unit tests for batch exit cooldown handling."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.execution.bracket_manager import BracketManager, BracketState


def test_fire_exits_batch_ignores_stale_cooldown(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify batch cooldown isolation. Args: monkeypatch. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        order_manager = MagicMock()
        manager = BracketManager(order_manager=order_manager)

        bracket = BracketState(
            entry_order_id='entry-1',
            symbol='NFO:NIFTY26SEP2500PE',
            side='BUY',
            quantity=5,
            entry_price=100.0,
            sl_trigger_price=95.0,
            tp_trigger_price=110.0,
        )
        bracket.last_ltp = 100.0

        action = {'type': 'SL', 'qty': 1, 'reason': 'test', 'price': 100.0}

        fired: list[tuple] = []

        def _fake_execute_exit_order(*args: object, **kwargs: object) -> None:
            fired.append(args)

        monkeypatch.setattr(manager, '_execute_exit_order', _fake_execute_exit_order)

        manager._exit_cooldowns[bracket.entry_order_id] = time.time()
        manager._fire_exits_batch([(bracket, action)])

        assert fired
    except Exception as e:
        logger.error('Failure in test_fire_exits_batch_ignores_stale_cooldown: %s', e)
        raise
