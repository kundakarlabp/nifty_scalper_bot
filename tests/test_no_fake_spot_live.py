"""Tests guarding against fake spot prices in LIVE mode."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest

from nifty_scalper_bot.core.app import (
    _SYNTHETIC_FALLBACK_SPOT,
    _is_live_execution_mode,
    _wait_for_live_spot_or_raise,
)


class _StubMDM:
    """Stand-in MarketDataManager exposing the helpers used in startup."""

    def __init__(
        self,
        *,
        ws_tick: dict[str, Any] | None = None,
        cached_price: float = 0.0,
    ) -> None:
        self._ws_tick = ws_tick
        self._cached_price = cached_price

    async def wait_for_fresh_spot_tick(
        self,
        symbol: str,
        *,
        timeout: float,
        max_age_seconds: float | None,
        require_ws: bool,
    ) -> dict[str, Any] | None:
        return dict(self._ws_tick) if self._ws_tick else None

    def get_latest_price(self, _symbol: str) -> float:
        return float(self._cached_price)

    def get_cached_ltp(
        self, _symbol: str, *, max_age_seconds: float | None, require_ws: bool
    ) -> float:
        return float(self._cached_price)


def _ctx(mdm: _StubMDM) -> SimpleNamespace:
    return SimpleNamespace(market_data_manager=mdm)


def test_live_mode_with_fresh_ws_tick_returns_real_price(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("ALLOW_SYNTHETIC_MARKET_DATA", raising=False)
    mdm = _StubMDM(ws_tick={"last_price": 24875.5, "source": "ws"})
    price = asyncio.run(
        _wait_for_live_spot_or_raise(_ctx(mdm), timeout=0.2, configured_mode="LIVE")
    )
    assert price == pytest.approx(24875.5)
    assert price != _SYNTHETIC_FALLBACK_SPOT


def test_live_mode_without_fresh_tick_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.delenv("ALLOW_SYNTHETIC_MARKET_DATA", raising=False)
    mdm = _StubMDM(ws_tick=None, cached_price=0.0)
    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(
            _wait_for_live_spot_or_raise(
                _ctx(mdm), timeout=0.1, configured_mode="LIVE"
            )
        )
    assert "spot_ltp_unavailable" in str(excinfo.value)


def test_live_mode_uses_rest_fallback_without_synthetic(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    mdm = _StubMDM(ws_tick=None, cached_price=24123.45)
    ctx = _ctx(mdm)
    with caplog.at_level("INFO"):
        price = asyncio.run(
            _wait_for_live_spot_or_raise(
                ctx, timeout=0.1, configured_mode="LIVE"
            )
        )
    assert price == pytest.approx(24123.45)
    assert price != _SYNTHETIC_FALLBACK_SPOT
    assert ctx.live_orders_armed is False
    assert "STARTUP_SPOT_REST_FALLBACK_USED" in caplog.text
    assert "STARTUP_SPOT_REST_FALLBACK_READY" in caplog.text
    assert "LIVE_SPOT_READY symbol=NSE:NIFTY price=24123.45 source=rest_fallback" not in caplog.text


def test_paper_mode_falls_back_to_synthetic_with_log(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    monkeypatch.delenv("ALLOW_SYNTHETIC_MARKET_DATA", raising=False)
    mdm = _StubMDM(ws_tick=None, cached_price=0.0)
    with caplog.at_level("WARNING"):
        price = asyncio.run(
            _wait_for_live_spot_or_raise(
                _ctx(mdm), timeout=0.1, configured_mode="PAPER"
            )
        )
    assert price == pytest.approx(_SYNTHETIC_FALLBACK_SPOT)
    assert any("SYNTHETIC_SPOT_USED" in record.message for record in caplog.records)


def test_paper_mode_uses_cached_price_when_available(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "PAPER")
    monkeypatch.delenv("ENABLE_LIVE", raising=False)
    mdm = _StubMDM(ws_tick=None, cached_price=24123.45)
    with caplog.at_level("INFO"):
        price = asyncio.run(
            _wait_for_live_spot_or_raise(
                _ctx(mdm), timeout=0.1, configured_mode="PAPER"
            )
        )
    assert price == pytest.approx(24123.45)
    assert "STARTUP_SPOT_FALLBACK_PROBE" in caplog.text


def test_allow_synthetic_market_data_overrides_live_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EXECUTION_MODE", "LIVE")
    monkeypatch.setenv("ALLOW_SYNTHETIC_MARKET_DATA", "true")
    mdm = _StubMDM(ws_tick=None, cached_price=0.0)
    price = asyncio.run(
        _wait_for_live_spot_or_raise(
            _ctx(mdm), timeout=0.1, configured_mode="LIVE"
        )
    )
    assert price == pytest.approx(_SYNTHETIC_FALLBACK_SPOT)


class TestIsLiveExecutionMode:
    def test_live_via_execution_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "LIVE")
        monkeypatch.delenv("ENABLE_LIVE", raising=False)
        assert _is_live_execution_mode() is True

    def test_live_via_enable_live_flag(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "PAPER")
        monkeypatch.setenv("ENABLE_LIVE", "true")
        assert _is_live_execution_mode() is True

    def test_paper_mode_is_not_live(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("EXECUTION_MODE", "PAPER")
        monkeypatch.setenv("ENABLE_LIVE", "false")
        assert _is_live_execution_mode() is False
