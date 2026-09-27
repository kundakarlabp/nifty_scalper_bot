from __future__ import annotations

import inspect

from nifty_scalper_bot.strategies import signal_generator


def test_base_strategy_manager_has_no_parallel_adx_regime_classifier() -> None:
    source = inspect.getsource(signal_generator.StrategyManager.generate_signal)
    assert "REGIME_ADX_TREND_MIN" not in source
    assert "REGIME_ADX_RANGE_MAX" not in source
    assert "_MEAN_REVERSION_TAGS" not in source
    assert "_TREND_TAGS" not in source
