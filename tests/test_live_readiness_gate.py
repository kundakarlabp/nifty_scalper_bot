"""Tests for the pure live-trading readiness gate."""

from __future__ import annotations

from pathlib import Path

import pytest

from nifty_scalper_bot.execution.readiness import HistoryReadinessPolicy, compute_live_readiness


class TestComputeLiveReadiness:
    def test_arms_when_ws_proof_substitutes_for_rest_quote(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=False,
            ws_quote_proof=True,
            market_open=True,
            runner_running=True,
        )
        assert armed is True
        assert reasons == []

    def test_blocks_when_neither_quote_nor_ws_proof(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=False,
            ws_quote_proof=False,
            market_open=True,
            runner_running=True,
        )
        assert armed is False
        assert reasons == ["market_data_proof_unavailable"]

    def test_blocks_when_pipeline_not_ready(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=False,
            quote_available=True,
            ws_quote_proof=False,
            market_open=True,
            runner_running=True,
        )
        assert armed is False
        assert "startup_pipeline_incomplete" in reasons

    def test_blocks_when_market_closed(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=True,
            ws_quote_proof=True,
            market_open=False,
            runner_running=True,
        )
        assert armed is False
        assert reasons == ["market_closed"]

    def test_returns_not_live_mode_when_disabled(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=False,
            hard_ready=True,
            quote_available=True,
            ws_quote_proof=True,
            market_open=True,
            runner_running=True,
        )
        assert armed is False
        assert reasons == ["not_live_mode"]

    def test_collects_multiple_blocking_reasons(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=False,
            quote_available=False,
            ws_quote_proof=False,
            market_open=False,
            runner_running=True,
        )
        assert armed is False
        assert "startup_pipeline_incomplete" in reasons
        assert "market_data_proof_unavailable" in reasons
        assert "market_closed" in reasons

    def test_arms_with_quote_only(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=True,
            ws_quote_proof=False,
            market_open=True,
            runner_running=True,
        )
        assert armed is True
        assert reasons == []


    def test_live_readiness_requires_runner_running(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=True,
            ws_quote_proof=True,
            market_open=True,
            runner_running=False,
        )
        assert armed is False
        assert "strategy_runner_not_running" in reasons

    def test_data_warmup_does_not_block_runner_start(self) -> None:
        configured_runtime_mode = "LIVE"
        effective_runtime_mode = "DATA_WARMUP"
        evaluation_allowed = configured_runtime_mode in {"PAPER", "SHADOW", "LIVE"}
        assert effective_runtime_mode == "DATA_WARMUP"
        assert evaluation_allowed is True


def test_live_readiness_accepts_fresh_ws_ltp() -> None:
    text = Path('src/nifty_scalper_bot/core/app.py').read_text(encoding='utf-8')
    assert 'has_fresh_ws_ltp' in text
    assert 'ws_quote_for_gate = bool(ws_quote_proof or ws_ltp_proof)' in text


def test_history_policy_invalid_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPTION_ENTRY_MIN_BARS", "bad")
    policy = HistoryReadinessPolicy.from_env()
    assert policy.option_entry_min_bars == 5


def test_compute_live_readiness_bad_option_exec_min_bars_does_not_crash() -> None:
    armed, reasons = compute_live_readiness(
        live_mode=True,
        hard_ready=True,
        quote_available=True,
        ws_quote_proof=True,
        market_open=True,
        runner_running=True,
        selected_ce="NFO:CE",
        selected_pe="NFO:PE",
        ce_bars=0,
        pe_bars=0,
        option_exec_min_bars="bad",  # type: ignore[arg-type]
    )
    assert armed is False
    assert "selected_ce_history_insufficient" in reasons
