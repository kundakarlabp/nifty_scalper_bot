from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_extract_regime_scale_honours_default() -> None:
    manager = StrategyManager([], None, None)
    assert manager._extract_regime_scale({}) == 1.0
