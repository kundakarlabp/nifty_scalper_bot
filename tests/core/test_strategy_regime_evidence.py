from __future__ import annotations

from nifty_scalper_bot.core.strategy_manager import RegimeState, StrategyManager


class _Strategy:
    def __init__(self, name: str) -> None:
        self.name = name
        self.config: dict[str, object] = {}

    def get_required_indicators(self) -> set[str]:
        return set()


def test_specific_regime_never_borrows_unknown_bucket() -> None:
    manager = StrategyManager([_Strategy("A")], None, None)
    manager.record_trade_result("A", 100.0, metadata={"regime": "unknown"})
    manager._regime_state = RegimeState(regime="trend", confidence=1.0)

    score = manager.get_strategy_scores()[0]

    assert score.active_regime_stats["regime"] == "trend"
    assert score.active_regime_stats["trades"] == 0.0
    assert score.active_regime_stats["evidence_fraction"] == 0.0


def test_sparse_regime_evidence_is_shrunk_toward_aggregate_performance() -> None:
    manager = StrategyManager([_Strategy("A"), _Strategy("B")], None, None)

    for _ in range(4):
        manager.record_trade_result("A", -100.0, metadata={"regime": "unknown"})
        manager.record_trade_result("B", 100.0, metadata={"regime": "unknown"})

    manager.record_trade_result("A", 100.0, metadata={"regime": "trend"})
    manager.record_trade_result("B", -100.0, metadata={"regime": "trend"})
    manager._regime_state = RegimeState(regime="trend", confidence=1.0)

    scores = {entry.strategy: entry for entry in manager.get_strategy_scores()}

    assert scores["A"].active_regime_stats["evidence_fraction"] == 0.2
    assert scores["B"].active_regime_stats["evidence_fraction"] == 0.2
    assert scores["A"].regime_score > scores["B"].regime_score
    assert scores["A"].score < scores["B"].score
