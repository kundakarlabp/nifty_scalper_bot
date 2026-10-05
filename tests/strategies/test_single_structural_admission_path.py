from __future__ import annotations

import inspect

from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_strategy_manager_has_single_post_combine_admission_path() -> None:
    source = inspect.getsource(StrategyManager.generate_signal)

    assert "self._filter_signal(combined)" not in source
    assert "orchestrator.filter_signal(" not in source
    assert 'metadata.get("kelly_fraction"' not in source
