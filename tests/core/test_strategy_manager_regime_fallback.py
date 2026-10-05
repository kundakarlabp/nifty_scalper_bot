from __future__ import annotations

import time

from nifty_scalper_bot.core.strategy_manager import (
    Signal,
    StrategyEvidence,
    StrategyManager,
    resolve_canonical_vwap,
    signal_to_evidence,
)


def test_extract_regime_scale_honours_default() -> None:
    manager = StrategyManager([], None, None)
    assert manager._extract_regime_scale({}) == 1.0


def test_record_trade_result_populates_passive_adaptive_statistics() -> None:
    manager = StrategyManager([], None, None)
    manager.record_trade_result("VWAPPro", 125.0, metadata={"regime": "trend"})
    stats = manager._adaptive_store.get_stats("VWAPPro")
    assert stats.win_rate == 1.0
    assert stats.avg_win == 125.0


def _signal(symbol: str, *, metadata: dict) -> Signal:
    return Signal(
        symbol=symbol,
        action="BUY",
        confidence=1.0,
        reason="test",
        quantity=65,
        stop_loss=90.0,
        take_profit=120.0,
        metadata=metadata,
    )


def test_evidence_boundary_stamps_canonical_strategy_identity() -> None:
    evidence = signal_to_evidence(
        _signal(
            "NFO:NIFTY24500CE",
            metadata={
                "setup_pass": True,
                "strategy_key": "wrong",
                "strategy_role": "wrong",
                "signal_family": "wrong",
            },
        ),
        "VWAPPro",
    )
    assert evidence.strategy == "VWAPPro"
    assert evidence.metadata["strategy_key"] == "vwap_pro"
    assert evidence.metadata["strategy_role"] == "trigger"
    assert evidence.metadata["signal_family"] == "reclaim_structure"
    assert evidence.metadata["setup_name"] == "test"


def test_evidence_boundary_prefers_explicit_setup_name() -> None:
    evidence = signal_to_evidence(
        _signal(
            "NFO:NIFTY24500CE",
            metadata={"setup_pass": True, "setup_name": "canonical_setup"},
        ),
        "VWAPPro",
    )
    assert evidence.metadata["setup_name"] == "canonical_setup"


def test_evidence_boundary_uses_taxonomy_for_context_aliases() -> None:
    evidence = signal_to_evidence(
        _signal("NFO:NIFTY24500CE", metadata={"role": "context"}),
        "OrderFlow",
    )
    assert evidence.metadata["strategy_key"] == "order_flow"
    assert evidence.metadata["strategy_role"] == "context"
    assert evidence.metadata["signal_family"] == "directional_context"


def test_contract_side_conflict_is_flagged_for_rejection() -> None:
    evidence = signal_to_evidence(
        _signal(
            "NFO:NIFTY24500CE",
            metadata={"setup_pass": True, "trade_side": "PE"},
        ),
        "ConflictedStrategy",
    )
    assert evidence.metadata["side_conflict"] is True
    assert evidence.metadata["side_from_metadata"] == "PE"
    assert evidence.metadata["no_vote_reason"] == "strategy_contract_side_conflict"


def test_undated_or_stale_context_cannot_confirm_entry() -> None:
    manager = StrategyManager([], None, None)
    current = StrategyEvidence(
        strategy="OrderFlow",
        side="CE",
        reasons=[],
        metadata={"role": "context", "vote_timestamp": time.time()},
    )
    undated = StrategyEvidence(
        strategy="OrderFlow",
        side="CE",
        reasons=[],
        metadata={"role": "context"},
    )
    stale = StrategyEvidence(
        strategy="OrderFlow",
        side="CE",
        reasons=[],
        metadata={"role": "context", "vote_timestamp": time.time() - 600.0},
    )
    assert manager._context_vote_is_timestamped(current) is True
    assert manager._context_vote_is_timestamped(undated) is False
    assert manager._context_vote_is_timestamped(stale) is False


def test_canonical_vwap_prefers_session_over_rolling_and_never_invents_one() -> None:
    assert (
        resolve_canonical_vwap(
            {"vwap": 102.94, "session_vwap": 103.31, "exchange_vwap": 103.45}
        )
        == 103.45
    )
    assert resolve_canonical_vwap({"vwap": 102.94, "session_vwap": 103.31}) == 103.31
    assert resolve_canonical_vwap({"vwap": 102.94}) == 102.94
    assert resolve_canonical_vwap({"current_price": 103.0, "ltp": 103.0}) is None
    assert resolve_canonical_vwap({}) is None
