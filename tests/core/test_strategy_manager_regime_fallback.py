from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import StrategyManager


def test_extract_regime_scale_honours_default() -> None:
    manager = StrategyManager([], None, None)
    assert manager._extract_regime_scale({}) == 1.0


def test_record_trade_result_populates_passive_adaptive_statistics() -> None:
    manager = StrategyManager([], None, None)

    manager.record_trade_result("VWAPPro", 125.0, metadata={"regime": "trend"})

    stats = manager._adaptive_store.get_stats("VWAPPro")
    assert stats.win_rate == 1.0
    assert stats.avg_win == 125.0
