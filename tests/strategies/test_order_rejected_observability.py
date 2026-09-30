"""ORDER_REJECTED must preserve its causal fields.

Post-#943 audit, P1. The warning carried only {"event", "symbol"}, so a
rejection could not be attributed to a safety gate, local validation,
margin/sizing or an actual broker failure, and could not be correlated with
the matching RUNNER_ORDER_MANAGER_REJECTED record. Three ORDER_REJECTED
events appeared in the 29 July session with no recoverable cause.

submit_reason, submit_details, broker_attempted and trace_id were all already
in scope at the emit site; they were simply discarded.
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

from nifty_scalper_bot.strategies.runner import StrategyRunner


def _emit_region() -> str:
    """Source around the ORDER_REJECTED emit site."""
    src = inspect.getsource(StrategyRunner)
    marker = '"ORDER_REJECTED by order_manager'
    assert marker in src, "ORDER_REJECTED emit site not found"
    start = src.index(marker)
    return src[start - 700 : start + 1400]


def test_order_rejected_carries_reason() -> None:
    """Without the reason a rejection class cannot be identified."""
    region = _emit_region()
    assert '"reason": submit_reason' in region


def test_order_rejected_carries_broker_attempted() -> None:
    """THE KEY DISCRIMINATOR: False means it never reached the broker."""
    region = _emit_region()
    assert '"broker_attempted": broker_attempted' in region


def test_order_rejected_carries_trace_id_for_correlation() -> None:
    """Needed to join with RUNNER_ORDER_MANAGER_REJECTED."""
    region = _emit_region()
    assert '"trace_id": trace_id' in region


def test_order_rejected_carries_structured_details() -> None:
    """Margin/sizing and kill-switch payloads must survive."""
    region = _emit_region()
    assert '"details": _reject_details' in region


def test_order_rejected_throttle_key_is_reason_scoped() -> None:
    """A 300s throttle keyed only on symbol hid distinct later causes."""
    region = _emit_region()
    assert 'f"runner_order_rejected_{base_symbol}_{submit_reason}"' in region


def test_order_rejected_still_identifies_both_symbols() -> None:
    """Underlying and the actual order symbol are both retained."""
    region = _emit_region()
    assert '"symbol": base_symbol' in region
    assert '"order_symbol": order_symbol' in region


def test_trade_decision_snapshot_is_persisted_to_existing_journal() -> None:
    captured: list[tuple[dict[str, object], str | None]] = []
    runner = SimpleNamespace(
        _runtime_live_orders_armed=True,
        _last_trade_decision=None,
        _order_manager=SimpleNamespace(
            record_trade_decision=lambda snapshot, trace_id=None: captured.append(
                (snapshot, trace_id)
            )
        ),
        _logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
    )

    StrategyRunner._record_trade_decision_snapshot(
        runner,
        symbol="NFO:NIFTYCE",
        direction="CE",
        final_reason="order_submitted",
        candidate_count=2,
        selected_candidate="NFO:NIFTYCE",
        strategy_allowed=True,
        risk_allowed=True,
        order_submitted=True,
        trace_id="trace-1",
    )

    assert captured[0][0]["candidate_count"] == 2
    assert captured[0][0]["order_submitted"] is True
    assert captured[0][1] == "trace-1"


def test_approved_decision_uses_executed_signal_identity_and_score() -> None:
    captured: list[dict[str, object]] = []
    runner = SimpleNamespace(
        _runtime_live_orders_armed=True,
        _last_trade_decision=None,
        _order_manager=SimpleNamespace(
            record_trade_decision=lambda snapshot, trace_id=None: captured.append(
                snapshot
            )
        ),
        _logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
    )

    StrategyRunner._record_trade_decision_snapshot(
        runner,
        symbol="NFO:NIFTYCE",
        direction="CE",
        final_reason="order_submitted",
        order_submitted=True,
        trace_id="runner-trace",
        signal_id="executed-signal",
        signal_score=7.8,
    )

    assert captured[0]["signal_id"] == "executed-signal"
    assert captured[0]["trade_id"] == "TRD_executed-signal"
    assert captured[0]["signal_score"] == 7.8


def test_decision_snapshot_persists_research_context() -> None:
    captured: list[dict[str, object]] = []
    runner = SimpleNamespace(
        _runtime_live_orders_armed=True,
        _last_trade_decision=None,
        _order_manager=SimpleNamespace(
            record_trade_decision=lambda snapshot, trace_id=None: captured.append(
                snapshot
            )
        ),
        _logger=SimpleNamespace(debug=lambda *_args, **_kwargs: None),
    )

    StrategyRunner._record_trade_decision_snapshot(
        runner,
        symbol="NFO:NIFTYCE",
        direction="CE",
        final_reason="alpha_below_threshold",
        order_submitted=False,
        trace_id="blocked-trace",
        signal_score=7.2,
        research_context={
            "strategy": "vwap_pro",
            "regime": "TREND",
            "signal_quality": {"final_score": 7.2, "alpha_score": 6.4},
            "rejection_stage": "runner_final_score",
        },
    )

    assert captured[0]["order_submitted"] is False
    assert captured[0]["research_context"] == {
        "strategy": "vwap_pro",
        "regime": "TREND",
        "signal_quality": {"final_score": 7.2, "alpha_score": 6.4},
        "rejection_stage": "runner_final_score",
    }


def test_margin_needed_rejection_is_deterministic_risk_capacity() -> None:
    assert (
        StrategyRunner._deterministic_execution_reject_reason("MARGIN needed=11225.50")
        == "risk_capacity_unavailable"
    )


def test_margin_no_qty_rejection_is_deterministic_risk_capacity() -> None:
    assert (
        StrategyRunner._deterministic_execution_reject_reason("margin_no_qty")
        == "risk_capacity_unavailable"
    )


def test_decision_research_context_does_not_fabricate_quality() -> None:
    context = StrategyRunner._decision_research_context(
        metadata={"strategy": "VWAPPro", "regime": "RANGE"},
        quality=None,
        stage="pre_score",
    )

    assert context["strategy"] == "VWAPPro"
    assert context["regime"] == "RANGE"
    assert context["decision_stage"] == "pre_score"
    assert "signal_quality" not in context


def test_decision_research_context_carries_known_quality() -> None:
    quality = SimpleNamespace(
        components={
            "strategy_name": "VWAPPro",
            "final_score": 7.4,
            "alpha_score": 6.9,
        }
    )
    context = StrategyRunner._decision_research_context(
        metadata={"strategy": "VWAPPro", "regime": "TREND", "approval_path": "x"},
        quality=quality,
        stage="execution",
    )

    assert context["signal_quality"]["final_score"] == 7.4
    assert context["decision_stage"] == "execution"
    assert context["approval_path"] == "x"


def test_reject_signal_execution_forwards_existing_research_context() -> None:
    captured: list[dict[str, object]] = []
    runner = SimpleNamespace(
        _emit_signal_execution_result=lambda **_kwargs: None,
        _record_trade_decision_snapshot=lambda **kwargs: captured.append(kwargs),
        _logger=SimpleNamespace(info=lambda *_args, **_kwargs: None),
    )
    details = {
        "direction": "PE",
        "signal_score": 7.3,
        "research_context": {
            "strategy": "VWAPPro",
            "decision_stage": "execution",
            "signal_quality": {"final_score": 7.3},
        },
    }

    StrategyRunner._reject_signal_execution(
        runner,
        symbol="NFO:NIFTYPE",
        trace_id="trace-provenance",
        reason="no_affordable_execution_candidate",
        details=details,
    )

    assert captured[0]["signal_score"] == 7.3
    assert captured[0]["research_context"] == details["research_context"]


def test_decision_research_context_preserves_existing_setup_provenance_only() -> None:
    metadata = {
        "strategy": "VWAPPro",
        "regime": "TREND",
        "setup_id": "vwap:CE:anchor:2026-09-30:NFO:NIFTYCE",
        "setup_name": "premium_vwap_reclaim",
        "strategy_key": "vwap_pro",
        "strategy_role": "trigger",
        "signal_family": "vwap",
        "contract_side": "CE",
        "raw_setup_score": 6.5,
        "score_contract_version": 1,
        "score_lineage": {"raw_setup_score": 6.5},
        "underlying_direction_bias": "CE",
        "underlying_direction_confidence": 0.91,
        "context_age_seconds": 0.8,
        "spread_pct": 0.32,
        "iv_rank": 48.0,
        "unrelated_runtime_object": object(),
    }

    context = StrategyRunner._decision_research_context(
        metadata=metadata,
        quality=None,
        stage="runner_final_score",
    )

    assert context["setup_id"] == metadata["setup_id"]
    assert context["raw_setup_score"] == 6.5
    assert context["underlying_direction_bias"] == "CE"
    assert context["spread_pct"] == 0.32
    assert "unrelated_runtime_object" not in context


def test_decision_research_context_does_not_invent_setup_identity() -> None:
    context = StrategyRunner._decision_research_context(
        metadata={"strategy": "ORBPro", "regime": "TREND"},
        quality=None,
        stage="manager",
    )

    assert "setup_id" not in context
    assert "setup_structure_id" not in context
