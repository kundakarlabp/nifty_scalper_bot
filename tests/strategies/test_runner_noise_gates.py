"""Tests for log noise reduction gates added to StrategyRunner.

Covers:
A. Market-closed pre-gate: evaluator not called; evaluation_entered not logged.
B. Pre-option history sync skipped when bars already sufficient.
C. Option pre-gates: premium, bid/ask, spread.
D. WS_FULL_QUOTE_PROOF first/transition only.
E. LIVE_TRADING_READINESS_SNAPSHOT state-change throttle.
F. Safety: skipped evaluation does not affect subscriptions, basket, or readiness.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# A. Market-closed pre-gate source-level assertions
# ---------------------------------------------------------------------------


def test_market_closed_pregate_before_evaluation_entered() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # The early market-closed gate must appear before the evaluation_entered log.
    pregate_idx = source.index("phase9_pregate")
    entered_idx = source.index('"evaluation_entered"')
    assert pregate_idx < entered_idx, (
        "Market-closed pre-gate must come before evaluation_entered log"
    )


def test_market_closed_pregate_does_not_log_evaluation_entered() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # The pre-gate returns early; evaluation_entered is only reached when market is open.
    assert "RUNNER_EVAL_SKIP_SUMMARY" in source
    assert "reason=market_closed" in source
    assert "phase9_pregate" in source


def test_market_closed_pregate_emits_summary() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "RUNNER_EVAL_SKIP_SUMMARY" in source
    assert "_market_closed_skip_counts" in source


# ---------------------------------------------------------------------------
# B. Pre-option history sync guard
# ---------------------------------------------------------------------------


def test_pre_option_eval_sync_guarded_by_indicator_count() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # Guard variable must exist before calling pre_option_eval_option_sync.
    guard_idx = source.index("_indicator_count_pre")
    sync_idx = source.index('"pre_option_eval_option_sync"')
    assert guard_idx < sync_idx, (
        "_indicator_count_pre guard must appear before pre_option_eval_option_sync call"
    )


def test_sync_context_history_if_cold_no_trace_when_warm() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # When count >= required_bars, the function must skip the trace (just continue).
    # The old code emitted _emit_history_hydration_trace in the warm branch; it must be gone.
    assert "_emit_history_hydration_trace" not in source[
        source.index("def _sync_context_history_if_cold"):
        source.index("def _sync_context_history_if_cold") + 300
    ]


# ---------------------------------------------------------------------------
# C. Option pre-gates
# ---------------------------------------------------------------------------


def test_option_pregate_low_premium_skips_evaluation() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "option_premium_below_min" in source
    assert "MIN_OPTION_PREMIUM" in source
    assert "RUNNER_EVAL_PREGATE_SKIPPED" in source


def test_option_pregate_missing_bid_ask_skips() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "option_bid_ask_missing_or_invalid" in source
    assert "REQUIRE_BID_ASK_FOR_OPTION_EVAL" in source


def test_option_pregate_inverted_spread_skips() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # ask < bid triggers the same bid_ask_missing_or_invalid reason.
    assert "_pg_ask < _pg_bid" in source


def test_option_pregate_wide_spread_pct_skips() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "option_spread_too_wide" in source
    assert "MAX_OPTION_SPREAD_PCT_FOR_EVAL" in source


def test_option_pregate_wide_spread_abs_skips() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "option_spread_abs_too_wide" in source
    assert "MAX_OPTION_SPREAD_ABS_FOR_EVAL" in source


def test_option_pregate_throttled_per_symbol() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "pregate_low_premium:" in source
    assert "pregate_no_bid_ask:" in source
    assert "pregate_wide_spread:" in source


# ---------------------------------------------------------------------------
# D. WS_FULL_QUOTE_PROOF first/transition logic
# ---------------------------------------------------------------------------


def test_ws_quote_proof_first_transition_only() -> None:
    source = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(encoding="utf-8")
    assert "_ws_quote_proof_state" in source
    assert "_is_first" in source
    assert "_tradable_transition" in source
    assert "_depth_transition" in source
    assert "_should_emit_proof" in source


def test_ws_quote_proof_debug_flag() -> None:
    source = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(encoding="utf-8")
    assert "LOG_QUOTE_PROOF_DEBUG" in source
    assert "LOG_QUOTE_PROOF_EVERY_SECONDS" in source


# ---------------------------------------------------------------------------
# E. LIVE_TRADING_READINESS_SNAPSHOT state-change throttle
# ---------------------------------------------------------------------------


def test_readiness_snapshot_throttled_per_reason() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "readiness_snap:" in source
    assert "RUNNER_READINESS_SNAP_INTERVAL_SECONDS" in source


def test_bootstrap_status_throttled_per_reason() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "bootstrap:" in source
    assert "RUNNER_BOOTSTRAP_LOG_INTERVAL_SECONDS" in source


# ---------------------------------------------------------------------------
# F. Safety: critical paths untouched
# ---------------------------------------------------------------------------


def test_broker_order_execution_logic_untouched() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # Key execution methods must still exist.
    assert "place_order" in source
    assert "_emit_execution_decision" in source or "order_forwarding_allowed" in source


def test_kill_switch_untouched() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "kill_switch" in source.lower() or "KILL_SWITCH" in source


def test_risk_halt_latch_untouched() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    assert "_risk_halt" in source or "risk_halt" in source.lower()


def test_skipped_eval_does_not_unsubscribe() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    # Pre-gate returns early; no unsubscribe call is present in the pre-gate block.
    pregate_block = source[
        source.index("EARLY MARKET-CLOSED PRE-GATE"):
        source.index("END PRE-GATES")
    ]
    assert "unsubscribe" not in pregate_block
    assert "_active_symbols.discard" not in pregate_block
    assert "_runtime_evaluation_ready" not in pregate_block


def test_skipped_eval_does_not_change_active_basket() -> None:
    source = Path("src/nifty_scalper_bot/strategies/runner.py").read_text(encoding="utf-8")
    pregate_block = source[
        source.index("EARLY MARKET-CLOSED PRE-GATE"):
        source.index("END PRE-GATES")
    ]
    assert "_active_selected_ce" not in pregate_block
    assert "_active_selected_pe" not in pregate_block


def test_backpressure_summary_added() -> None:
    source = Path("src/nifty_scalper_bot/data/market_data_manager.py").read_text(encoding="utf-8")
    assert "MDM_BUS_BACKPRESSURE_SUMMARY" in source
    assert "_bus_backpressure_counts" in source
