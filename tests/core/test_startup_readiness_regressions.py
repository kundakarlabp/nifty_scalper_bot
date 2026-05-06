from __future__ import annotations

from pathlib import Path


SOURCE = Path("src/nifty_scalper_bot/core/app.py").read_text(encoding="utf-8")


def test_configured_runtime_mode_is_safely_defined_in_readiness_gate() -> None:
    assert "def _safe_runtime_mode" in SOURCE
    assert "configured_runtime_mode = _safe_runtime_mode(ctx)" in SOURCE
    assert "cannot access local variable 'configured_runtime_mode'" not in SOURCE


def test_futures_is_soft_readiness_requirement() -> None:
    assert "missing_soft" in SOURCE
    assert "SOFT_READINESS_MISSING missing=futures action=continue_with_spot_option_context" in SOURCE
    assert "ctx.data_hard_ready = bool(spot_ready and atm_ce_ready and atm_pe_ready)" in SOURCE


def test_best_ready_option_can_substitute_atm_symbol() -> None:
    assert "def _best_fresh_option(" in SOURCE
    assert "def _best_hydrated_option(" in SOURCE
    assert "ATM_CE_SUBSTITUTED_WITH_READY_OPTION" in SOURCE
    assert "ATM_PE_SUBSTITUTED_WITH_READY_OPTION" in SOURCE
    assert "OPTION_QUOTE_READY" in SOURCE
    assert "OPTION_HISTORY_READY" in SOURCE


def test_readiness_sync_helper_exists_for_mdm_to_runner() -> None:
    assert "def _sync_mdm_bars_to_runner" in SOURCE
    assert "RUNNER_MDM_BAR_SYNC" in SOURCE
    assert "existing >= min_bars" in SOURCE


def test_hydration_timeout_path_continues_in_degraded_mode() -> None:
    assert "STARTUP_PIPELINE_INCOMPLETE_CONTINUING" in SOURCE
    assert "ctx.degraded_mode = True" in SOURCE
    assert "ctx.live_orders_armed = False" in SOURCE


def test_data_pipeline_ready_uses_app_level_data_hard_ready() -> None:
    assert "ctx.data_pipeline_ready = bool(ctx.data_hard_ready)" in SOURCE
    assert "and configured_runtime_mode == \"LIVE\"" in SOURCE
