from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

from nifty_scalper_bot.core.strategy_manager import (
    Signal,
    StrategyEvidence,
    StrategyManager,
    _enrich_smc_pre_strategy,
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


def test_canonical_vwap_rejects_nonfinite_values_before_fallback() -> None:
    assert (
        resolve_canonical_vwap(
            {"exchange_vwap": float("inf"), "session_vwap": 103.31, "vwap": 102.94}
        )
        == 103.31
    )
    assert (
        resolve_canonical_vwap(
            {"exchange_vwap": float("nan"), "session_vwap": -1.0, "vwap": 102.94}
        )
        == 102.94
    )
    assert (
        resolve_canonical_vwap(
            {"exchange_vwap": float("inf"), "session_vwap": float("nan"), "vwap": 0}
        )
        is None
    )


def test_smc_enrichment_rejects_unfinished_and_invalid_option_bars() -> None:
    clean = [
        {
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 100.0,
            "is_complete": True,
        }
        for _ in range(25)
    ]
    partial = {
        "open": 100.0,
        "high": 150.0,
        "low": 99.0,
        "close": 149.0,
        "volume": 3000.0,
        "is_complete": False,
    }
    invalid = {
        "open": 100.0,
        "high": float("nan"),
        "low": 99.0,
        "close": 140.0,
        "volume": 100.0,
    }
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE", {}, [*clean, partial, invalid]
    )
    assert enriched["smc_enrichment_bars"] == 25
    assert enriched["premium_current"] == 100.0
    assert enriched["bos_confirmed"] is False
    assert enriched["smc_enrichment_lookahead_safe"] is True


def test_smc_enrichment_drops_provisional_bar_without_complete_flag() -> None:
    history = [
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 100}
        for _ in range(25)
    ]
    history.append(
        {
            "open": 100.0,
            "high": 160.0,
            "low": 99.0,
            "close": 159.0,
            "volume": 1000,
            "is_provisional": True,
        }
    )
    enriched = _enrich_smc_pre_strategy("NFO:NIFTY26O1922700CE", {}, history)
    assert enriched["smc_enrichment_bars"] == 25
    assert enriched["premium_current"] == 100.0


def test_smc_option_enrichment_retains_completed_event_timestamps() -> None:
    start = datetime(2026, 10, 14, 5, 0, tzinfo=timezone.utc)
    history = [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 100.0,
            "is_complete": True,
        }
        for index in range(25)
    ]
    history[-2]["close"] = 90.0
    history[-1].update(
        {
            "timestamp": (start + timedelta(minutes=24)).isoformat(),
            "open": 90.0,
            "high": 112.0,
            "low": 89.0,
            "close": 110.0,
        }
    )
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE",
        {"session_vwap": 100.0, "direction_bias": "CE"},
        history,
    )
    expected = (start + timedelta(minutes=24)).timestamp()

    assert enriched["premium_reclaim"] is True
    assert enriched["premium_reclaim_ts_epoch"] == expected
    assert enriched["bos_confirmed"] is True
    assert enriched["bos_ts_epoch"] == expected


def test_smc_missing_bar_timestamps_never_invent_event_provenance() -> None:
    history = [
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 100}
        for _ in range(25)
    ]
    history[-2]["close"] = 90.0
    history[-1].update({"open": 90.0, "high": 112.0, "close": 110.0})
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE", {"session_vwap": 100.0}, history
    )
    assert enriched["premium_reclaim"] is True
    assert enriched["premium_reclaim_ts_epoch"] is None
    assert enriched["bos_ts_epoch"] is None


def test_smc_option_vwap_proxy_excludes_previous_session_bars() -> None:
    today = datetime(2026, 10, 14, 5, 0, tzinfo=timezone.utc)
    previous_day = today - timedelta(days=1)
    history = [
        {
            "timestamp": previous_day + timedelta(minutes=index),
            "open": 180.0,
            "high": 200.0,
            "low": 170.0,
            "close": 190.0,
            "volume": 100.0,
        }
        for index in range(23)
    ]
    history.extend(
        [
            {
                "timestamp": today,
                "open": 90.0,
                "high": 98.0,
                "low": 87.0,
                "close": 90.0,
                "volume": 100.0,
            },
            {
                "timestamp": today + timedelta(minutes=1),
                "open": 90.0,
                "high": 112.0,
                "low": 89.0,
                "close": 110.0,
                "volume": 100.0,
            },
        ]
    )
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE", {}, history
    )
    assert enriched["premium_vwap"] is not None
    assert 90 < enriched["premium_vwap"] < 110


def test_smc_no_timestamp_does_not_invent_proxy_vwap() -> None:
    history = [
        {"open": 100.0, "high": 102.0, "low": 98.0, "close": 100.0, "volume": 100}
        for _ in range(25)
    ]
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE", {}, history
    )
    assert enriched["premium_vwap"] is None
    assert enriched["premium_reclaim"] is False


def test_smc_does_not_invent_event_timestamp_for_external_structure_flag() -> None:
    start = datetime(2026, 10, 14, 5, 0, tzinfo=timezone.utc)
    history = [
        {
            "timestamp": start + timedelta(minutes=index),
            "open": 100.0,
            "high": 102.0,
            "low": 98.0,
            "close": 100.0,
            "volume": 100.0,
        }
        for index in range(25)
    ]
    history[-1].update({"open": 101.0, "high": 110.0, "close": 109.0})
    enriched = _enrich_smc_pre_strategy(
        "NFO:NIFTY26O1922700CE", {"bos_confirmed": True}, history
    )
    assert enriched["bos_confirmed"] is True
    assert enriched.get("bos_ts_epoch") is None
