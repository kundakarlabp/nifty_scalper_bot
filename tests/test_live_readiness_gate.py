"""Tests for the pure live-trading readiness gate."""

from __future__ import annotations

import pytest

from nifty_scalper_bot.execution.readiness import compute_live_readiness


class TestComputeLiveReadiness:
    def test_arms_when_ws_proof_substitutes_for_rest_quote(self) -> None:
        armed, reasons = compute_live_readiness(
            live_mode=True,
            hard_ready=True,
            quote_available=False,
            ws_quote_proof=True,
            market_open=True,
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
        )
        assert armed is True
        assert reasons == []
