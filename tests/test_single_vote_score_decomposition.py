from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_single_vote_score_log_path_available():
    assert hasattr(StrategyManager, '_emit_single_vote_decision')
