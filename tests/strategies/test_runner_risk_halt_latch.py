from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock


def test_risk_halt_latch_short_circuits_strategy_work(
    monkeypatch,
) -> None:
    """Verify risk-halt latch blocks redundant strategy processing.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.

    Raises:
        Exception: If setup stubs fail unexpectedly.
    """

    from nifty_scalper_bot.strategies.runner import StrategyRunner, StrategyRunnerConfig

    market_data = MagicMock()
    indicator = MagicMock()
    strategy_manager = MagicMock()
    risk_manager = MagicMock()
    order_manager = MagicMock()
    position_manager = MagicMock()
    bracket_manager = MagicMock()

    risk_manager.is_circuit_breaker_tripped.return_value = (True, 'dd_limit')

    runner = StrategyRunner(
        market_data_manager=market_data,
        indicator_engine=indicator,
        strategy_manager=strategy_manager,
        risk_manager=risk_manager,
        order_manager=order_manager,
        position_manager=position_manager,
        bracket_manager=bracket_manager,
        config=StrategyRunnerConfig(max_trade_history=4, min_indicator_bars=0),
        data_hub=None,
        strike_selector=None,
    )
    monkeypatch.setattr(runner, '_is_market_open', lambda _now: True)

    symbol = 'NFO:NIFTY2621025700PE'
    tick = {
        'ltp': 101.0,
        'timestamp': datetime.now(timezone.utc),
        'volume': 10,
        'source': 'ws',
    }

    runner._on_tick(symbol, tick)
    runner._on_tick(symbol, tick)

    assert runner._risk_halt_active is True
    assert risk_manager.is_circuit_breaker_tripped.call_count == 1
    assert bracket_manager.on_tick.call_count == 2
    assert strategy_manager.generate_signals.call_count == 0
    assert indicator.update_price.call_count == 0
    assert position_manager.update_position_price.call_count == 2
