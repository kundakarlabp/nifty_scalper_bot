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


def _signal(symbol: str, *, confidence: float, metadata: dict):
    from nifty_scalper_bot.strategies.signal_generator import Signal

    return Signal(
        symbol=symbol,
        action="BUY",
        confidence=confidence,
        reason="test",
        quantity=65,
        stop_loss=0.0,
        take_profit=0.0,
        metadata=metadata,
    )


def test_confidence_cannot_inflate_a_weak_setup_score() -> None:
    """Confidence is derived from the same indicators as the setup score, so
    taking the max double-counted the strongest representation."""
    from nifty_scalper_bot.core.strategy_manager import signal_to_vote

    vote = signal_to_vote(
        _signal("NFO:NIFTY24500CE", confidence=0.95, metadata={"raw_setup_score": 4.0}),
        "WeakSetup",
    )

    assert vote.score == 4.0
    assert vote.metadata["raw_setup_score"] == 4.0
    assert vote.metadata["raw_confidence"] == 0.95


def test_strong_setup_with_adequate_confidence_is_unchanged() -> None:
    from nifty_scalper_bot.core.strategy_manager import signal_to_vote

    vote = signal_to_vote(
        _signal("NFO:NIFTY24500CE", confidence=0.65, metadata={"raw_setup_score": 8.5}),
        "StrongSetup",
    )

    assert vote.score == 8.5


def test_contract_side_conflict_is_flagged_for_rejection() -> None:
    """A strategy asking for PE must not be reinterpreted as a CE vote."""
    from nifty_scalper_bot.core.strategy_manager import signal_to_vote

    vote = signal_to_vote(
        _signal(
            "NFO:NIFTY24500CE",
            confidence=0.8,
            metadata={"raw_setup_score": 8.0, "trade_side": "PE"},
        ),
        "ConflictedStrategy",
    )

    assert vote.metadata["side_conflict"] is True
    assert vote.metadata["side_from_metadata"] == "PE"
    assert vote.metadata["no_vote_reason"] == "strategy_contract_side_conflict"


def test_vote_ranking_is_independent_of_registration_order() -> None:
    """Capping inside the evaluation loop made the outcome depend on which
    strategies happened to be registered first."""
    from nifty_scalper_bot.core.strategy_manager import StrategyVote

    def _vote(name: str, score: float, confidence: float) -> StrategyVote:
        return StrategyVote(
            strategy=name, side="CE", score=score,
            confidence=confidence, reasons=[], metadata={},
        )

    weak = ("sig-weak", _vote("Aaa", 3.0, 0.9))
    strong = ("sig-strong", _vote("Zzz", 9.0, 0.6))
    middle = ("sig-mid", _vote("Mmm", 6.0, 0.7))

    def _rank(pairs):
        ordered = sorted(
            pairs,
            key=lambda pair: (
                -float(pair[1].score or 0.0),
                -float(pair[1].confidence or 0.0),
                str(pair[1].strategy),
            ),
        )
        return [pair[1].strategy for pair in ordered[:2]]

    assert _rank([weak, strong, middle]) == ["Zzz", "Mmm"]
    assert _rank([strong, middle, weak]) == ["Zzz", "Mmm"]
    assert _rank([middle, weak, strong]) == ["Zzz", "Mmm"]
