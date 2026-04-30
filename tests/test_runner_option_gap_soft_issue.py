from __future__ import annotations

import time

from nifty_scalper_bot.strategies.runner import StrategyRunner, SymbolState


def test_option_with_recent_tick_and_gap_not_degraded() -> None:
    runner = StrategyRunner()
    symbol = 'NFO:NIFTY26MAY25500PE'
    runner._required_candles = 3
    runner._session_gap_count[symbol] = 2
    runner._last_tick_time_by_symbol[symbol] = time.time()
    runner._has_session_candle_gaps = lambda _s: True  # type: ignore[assignment]

    state = runner.update_symbol_hydration(
        symbol,
        bars=[1.0, 2.0, 3.0],
        indicators={symbol: {'vwap': 200.0, 'cum_volume': 1000.0}},
    )
    assert state != SymbolState.DEGRADED
