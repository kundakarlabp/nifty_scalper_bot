from __future__ import annotations

import pytest

from nifty_scalper_bot.core.strategy_manager import StrategyManager


class _Strategy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config = {}

    def get_required_indicators(self) -> set[str]:
        return set()


def _manager() -> StrategyManager:
    manager = StrategyManager(
        [_Strategy("Alpha"), _Strategy("Beta")],
        None,
        None,
        regime_signal_getter=lambda: {"regime": "trend", "confidence": 1.0},
    )
    manager.configure_score_thresholds(min_trades=5)
    return manager


def test_sparse_performance_is_neutral_until_existing_evidence_floor() -> None:
    manager = _manager()
    manager.record_trade_result(
        "Alpha",
        100.0,
        metadata={"regime": "trend"},
    )

    scores = {entry.strategy: entry for entry in manager.get_strategy_scores()}

    assert scores["Alpha"].score == pytest.approx(0.5)
    assert scores["Beta"].score == pytest.approx(0.5)
    assert scores["Alpha"].regime_score == pytest.approx(0.5)
    assert scores["Beta"].regime_score == pytest.approx(0.5)


def test_unknown_history_never_substitutes_for_known_active_regime() -> None:
    manager = _manager()
    for _ in range(5):
        manager.record_trade_result("Alpha", 10.0, metadata={"regime": ""})
        manager.record_trade_result("Beta", 10.0, metadata={"regime": ""})

    scores = {entry.strategy: entry for entry in manager.get_strategy_scores()}

    assert scores["Alpha"].active_regime_stats["trades"] == 0.0
    assert scores["Beta"].active_regime_stats["trades"] == 0.0
    assert scores["Alpha"].regime_score == pytest.approx(0.5)
    assert scores["Beta"].regime_score == pytest.approx(0.5)
    assert scores["Alpha"].regime_breakdown["unknown"]["trades"] == 5.0


def test_exact_active_regime_contributes_after_evidence_floor() -> None:
    manager = _manager()
    for _ in range(5):
        manager.record_trade_result("Alpha", 10.0, metadata={"regime": "trend"})
        manager.record_trade_result("Beta", -10.0, metadata={"regime": "trend"})

    scores = {entry.strategy: entry for entry in manager.get_strategy_scores()}

    assert scores["Alpha"].active_regime_stats["trades"] == 5.0
    assert scores["Beta"].active_regime_stats["trades"] == 5.0
    assert scores["Alpha"].regime_score > scores["Beta"].regime_score
