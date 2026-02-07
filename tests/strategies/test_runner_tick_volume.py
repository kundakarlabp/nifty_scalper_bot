from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig


def test_on_tick_uses_last_quantity_volume(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify volume fallback. Args: monkeypatch. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        market_data = MagicMock()
        indicator = MagicMock()
        strategy_manager = MagicMock()
        risk_manager = MagicMock()
        order_manager = MagicMock()
        position_manager = MagicMock()

        risk_manager.can_trade.return_value = True

        runner = StrategyRunner(
            market_data_manager=market_data,
            indicator_engine=indicator,
            strategy_manager=strategy_manager,
            risk_manager=risk_manager,
            order_manager=order_manager,
            position_manager=position_manager,
            config=StrategyRunnerConfig(max_trade_history=4, min_indicator_bars=0),
            data_hub=None,
            strike_selector=None,
        )

        symbol = 'NFO:NIFTY2621025700PE'
        runner._startup_timestamp = time.time()
        monkeypatch.setattr(runner, '_is_market_open', lambda _now: True)

        builder = MagicMock()
        builder.update.return_value = None
        runner._bar_builders[symbol] = builder

        tick = {
            'ltp': 100.0,
            'last_quantity': 7,
            'timestamp': datetime.now(timezone.utc),
        }

        runner._on_tick(symbol, tick)

        builder.update.assert_called()
        assert builder.update.call_args[0][1] == 7
    except Exception as e:
        logger.error('Failure in test_on_tick_uses_last_quantity_volume: %s', e)
        raise


def test_on_tick_uses_order_book_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify book fallback. Args: monkeypatch. Returns: None. Raises: Exception."""
    logger = logging.getLogger(__name__)
    try:
        market_data = MagicMock()
        indicator = MagicMock()
        strategy_manager = MagicMock()
        risk_manager = MagicMock()
        order_manager = MagicMock()
        position_manager = MagicMock()

        risk_manager.can_trade.return_value = True

        runner = StrategyRunner(
            market_data_manager=market_data,
            indicator_engine=indicator,
            strategy_manager=strategy_manager,
            risk_manager=risk_manager,
            order_manager=order_manager,
            position_manager=position_manager,
            config=StrategyRunnerConfig(max_trade_history=4, min_indicator_bars=0),
            data_hub=None,
            strike_selector=None,
        )

        symbol = 'NFO:NIFTY2621025700PE'
        runner._startup_timestamp = time.time()
        monkeypatch.setattr(runner, '_is_market_open', lambda _now: True)

        builder = MagicMock()
        builder.update.return_value = None
        runner._bar_builders[symbol] = builder

        tick = {
            'ltp': 0.0,
            'best_bid': 100.0,
            'best_ask': 102.0,
            'timestamp': datetime.now(timezone.utc),
        }

        runner._on_tick(symbol, tick)

        builder.update.assert_called()
        assert builder.update.call_args[0][0] == pytest.approx(101.0)
    except Exception as e:
        logger.error('Failure in test_on_tick_uses_order_book_price: %s', e)
        raise
