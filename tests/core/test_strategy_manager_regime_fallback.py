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


def _context_vote(side: str, veto: float, *, stamped: bool = True):
    import time

    from nifty_scalper_bot.core.strategy_manager import StrategyVote

    metadata = {"context_veto_score": veto}
    if stamped:
        metadata["vote_timestamp"] = time.time()
    return StrategyVote(
        strategy="OrderFlow", side=side, score=veto,
        confidence=0.5, reasons=[], metadata=metadata,
    )


def test_undated_context_vote_cannot_hard_veto() -> None:
    """Defaulting a missing vote timestamp to 'now' let a provenance-free vote
    block a valid trade at full strength."""
    manager = StrategyManager([], None, None)

    assert manager._extract_context_veto_score(_context_vote("PE", 9.0)) == 9.0
    assert (
        manager._extract_context_veto_score(_context_vote("PE", 9.0, stamped=False))
        == 0.0
    )


def test_stale_context_vote_cannot_hard_veto() -> None:
    import time

    manager = StrategyManager([], None, None)
    vote = _context_vote("PE", 9.0)
    vote.metadata["vote_timestamp"] = time.time() - 600.0

    assert manager._extract_context_veto_score(vote) == 0.0


def test_canonical_vwap_prefers_session_over_rolling_and_never_invents_one() -> None:
    from nifty_scalper_bot.core.strategy_manager import resolve_canonical_vwap

    assert resolve_canonical_vwap(
        {"vwap": 102.94, "session_vwap": 103.31, "exchange_vwap": 103.45}
    ) == 103.45
    assert resolve_canonical_vwap({"vwap": 102.94, "session_vwap": 103.31}) == 103.31
    assert resolve_canonical_vwap({"vwap": 102.94}) == 102.94
    # No VWAP evidence must stay absent, never fall back to the current price.
    assert resolve_canonical_vwap({"current_price": 103.0, "ltp": 103.0}) is None
    assert resolve_canonical_vwap({}) is None


def test_quality_score_awards_nothing_for_missing_evidence() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyVote

    manager = StrategyManager([], None, None)
    bare = StrategyVote(
        strategy="Bare", side="CE", score=6.0, confidence=0.7,
        reasons=[], metadata={"raw_setup_score": 6.0},
    )

    _score, meta = manager._compute_trade_quality_score(
        bare, {}, symbol="NFO:NIFTY24500CE",
        selected_ok=True, near_atm_ok=True, context_votes=[],
    )
    components = meta["trade_quality_components"]

    assert components["direction_alignment"] == 0.0
    assert components["liquidity_spread_quality"] == 0.0
    assert components["freshness_tick_quality"] == 0.0
    assert components["market_regime_time_suitability"] == 0.0
    assert meta["quality_evidence_complete"] is False


def test_quality_score_credits_demonstrated_evidence() -> None:
    from nifty_scalper_bot.core.strategy_manager import StrategyVote

    manager = StrategyManager([], None, None)
    proven = StrategyVote(
        strategy="Proven", side="CE", score=6.0, confidence=0.7, reasons=[],
        metadata={
            "raw_setup_score": 6.0,
            "direction_alignment_score": 1.0,
            "liquidity_score": 2.0,
            "regime_time_suitability_score": 1.0,
        },
    )

    _score, meta = manager._compute_trade_quality_score(
        proven, {"stale_data_used": False}, symbol="NFO:NIFTY24500CE",
        selected_ok=True, near_atm_ok=True, context_votes=[],
    )
    components = meta["trade_quality_components"]

    assert components["direction_alignment"] == 1.0
    assert components["liquidity_spread_quality"] == 2.0
    assert components["freshness_tick_quality"] == 1.0
    assert meta["quality_evidence_complete"] is True
