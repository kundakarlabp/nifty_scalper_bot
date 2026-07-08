from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.core import app
from nifty_scalper_bot.core.app import _polling_fallback_degraded


def test_spot_stale_only_does_not_activate_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=True,
            options_fresh=True,
            quote_stale_ms=120000,
            feed_health={"spot_age_ms": 10**9},
            data_age_ms=10**9,
        )
        is False
    )


def test_futures_stale_requires_age_threshold_when_websocket_healthy() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=False,
            options_fresh=True,
            quote_stale_ms=120000,
            feed_health={"futures_age_ms": 70},
            data_age_ms=70,
        )
        is False
    )
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=False,
            options_fresh=True,
            quote_stale_ms=120000,
            feed_health={"futures_age_ms": 120000},
            data_age_ms=120000,
        )
        is True
    )


def test_options_stale_requires_selected_age_threshold_when_websocket_healthy() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=True,
            options_fresh=False,
            quote_stale_ms=120000,
            feed_health={"selected_ce_age_ms": 70, "selected_pe_age_ms": 60},
            data_age_ms=70,
        )
        is False
    )
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=False,
            futures_fresh=True,
            options_fresh=False,
            quote_stale_ms=120000,
            feed_health={"selected_ce_age_ms": 120000, "selected_pe_age_ms": 60},
            data_age_ms=120000,
        )
        is True
    )


def test_ws_disconnected_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=False,
            lagging=False,
            futures_fresh=True,
            options_fresh=True,
            quote_stale_ms=120000,
            data_age_ms=10,
        )
        is True
    )


def test_lagging_event_loop_activates_poll_fallback() -> None:
    assert (
        _polling_fallback_degraded(
            ws_ok=True,
            lagging=True,
            futures_fresh=True,
            options_fresh=True,
            quote_stale_ms=120000,
            data_age_ms=10,
        )
        is True
    )


class _Fallback:
    def __init__(self, *, running: bool = False) -> None:
        self.start = MagicMock()
        self.stop = MagicMock()
        self.set_websocket_mode = MagicMock()
        self._running = running

    def is_running(self) -> bool:
        return self._running


def _supervisor_ctx(
    *,
    is_connected: object,
    feed_health: dict[str, object] | None = None,
    data_age_ms: int = 10**9,
) -> SimpleNamespace:
    health = feed_health or {
        "futures_fresh": False,
        "options_fresh": False,
        "spot_fresh": False,
        "spot_symbol": "NSE:NIFTY",
        "spot_age_ms": 10**9,
    }
    return SimpleNamespace(
        websocket_manager=SimpleNamespace(is_connected=is_connected),
        market_data_manager=SimpleNamespace(
            trading_feed_health=MagicMock(return_value=health),
            data_age_ms=MagicMock(return_value=data_age_ms),
            ensure_spot_reference_fresh=MagicMock(),
        ),
    )


@pytest.mark.asyncio
async def test_offmarket_supervisor_stops_fallback_without_reading_feed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: False)
    ctx = _supervisor_ctx(is_connected=MagicMock())
    fallback = _Fallback(running=True)

    await app._polling_failover_supervisor_iteration(
        ctx,
        fallback,
        quote_stale_ms=1000,
        degraded_since=0.0,
        recovered_since=None,
        activate_after=0.0,
    )

    ctx.websocket_manager.is_connected.assert_not_called()
    ctx.market_data_manager.trading_feed_health.assert_not_called()
    ctx.market_data_manager.data_age_ms.assert_not_called()
    fallback.start.assert_not_called()
    fallback.set_websocket_mode.assert_called_once_with(True)
    fallback.stop.assert_called_once()


@pytest.mark.asyncio
async def test_stale_futures_activates_polling_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(
        is_connected=lambda: True,
        data_age_ms=120000,
        feed_health={
            "futures_fresh": False,
            "futures_age_ms": 120000,
            "options_fresh": True,
            "spot_fresh": True,
            "spot_symbol": "NSE:NIFTY",
            "spot_age_ms": 100,
        },
    )
    fallback = _Fallback()

    await app._polling_failover_supervisor_iteration(
        ctx,
        fallback,
        quote_stale_ms=1000,
        degraded_since=0.0,
        recovered_since=None,
        activate_after=0.0,
    )

    fallback.set_websocket_mode.assert_called_once_with(False)
    fallback.start.assert_called_once()


@pytest.mark.asyncio
async def test_noncallable_websocket_state_is_handled(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(is_connected=(False,))
    fallback = _Fallback()

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.core.app"):
        await app._polling_failover_supervisor_iteration(
            ctx,
            fallback,
            quote_stale_ms=1000,
            degraded_since=0.0,
            recovered_since=None,
            activate_after=0.0,
        )

    assert "POLLING_SUPERVISOR_NONCALLABLE" in caplog.text
    assert "websocket_manager.is_connected" in caplog.text


@pytest.mark.asyncio
async def test_noncallable_market_session_state_is_handled(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", (False,))
    ctx = _supervisor_ctx(is_connected=False)
    fallback = _Fallback()

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.core.app"):
        await app._polling_failover_supervisor_iteration(
            ctx,
            fallback,
            quote_stale_ms=1000,
            degraded_since=0.0,
            recovered_since=None,
            activate_after=0.0,
        )

    assert "POLLING_SUPERVISOR_NONCALLABLE" in caplog.text
    assert "is_market_open_now" in caplog.text
    assert "Failure in polling failover supervisor" not in caplog.text


@pytest.mark.asyncio
async def test_options_stale_but_age_below_threshold_does_not_start_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(
        is_connected=MagicMock(return_value=True),
        data_age_ms=70,
        feed_health={
            "futures_fresh": True,
            "options_fresh": False,
            "options_age_ms": 70,
            "spot_fresh": True,
            "spot_symbol": "NSE:NIFTY",
            "spot_age_ms": 70,
        },
    )
    fallback = _Fallback(running=False)

    await app._polling_failover_supervisor_iteration(
        ctx, fallback, quote_stale_ms=120000, degraded_since=0.0, recovered_since=None, activate_after=0.0
    )

    fallback.start.assert_not_called()


@pytest.mark.asyncio
async def test_options_stale_at_threshold_starts_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(
        is_connected=MagicMock(return_value=True),
        data_age_ms=120000,
        feed_health={
            "futures_fresh": True,
            "options_fresh": False,
            "options_age_ms": 120000,
            "spot_fresh": True,
            "spot_symbol": "NSE:NIFTY",
            "spot_age_ms": 70,
        },
    )
    fallback = _Fallback(running=False)

    await app._polling_failover_supervisor_iteration(
        ctx, fallback, quote_stale_ms=120000, degraded_since=0.0, recovered_since=None, activate_after=0.0
    )

    fallback.start.assert_called_once()


@pytest.mark.asyncio
async def test_missing_quote_or_websocket_failure_allows_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(
        is_connected=MagicMock(return_value=False),
        data_age_ms=10,
        feed_health={
            "futures_fresh": True,
            "options_fresh": True,
            "spot_fresh": True,
            "spot_symbol": "NSE:NIFTY",
            "spot_age_ms": 10,
        },
    )
    fallback = _Fallback(running=False)

    await app._polling_failover_supervisor_iteration(
        ctx, fallback, quote_stale_ms=120000, degraded_since=0.0, recovered_since=None, activate_after=0.0
    )

    fallback.start.assert_called_once()


@pytest.mark.asyncio
async def test_fallback_health_exception_logs_without_raising(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)

    def bad_health() -> dict[str, object]:
        raise RuntimeError("health boom")

    ctx = SimpleNamespace(
        websocket_manager=SimpleNamespace(is_connected=MagicMock(return_value=True)),
        market_data_manager=SimpleNamespace(
            trading_feed_health=bad_health,
            data_age_ms=MagicMock(return_value=70),
        ),
    )
    fallback = _Fallback(running=False)

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.core.app"):
        await app._polling_failover_supervisor_iteration(
            ctx, fallback, quote_stale_ms=120000, degraded_since=0.0, recovered_since=None, activate_after=0.0
        )

    assert "POLLING_FALLBACK_HEALTH_FAILED" in caplog.text
    fallback.start.assert_not_called()


@pytest.mark.asyncio
async def test_fallback_start_exception_is_nonfatal(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(app, "is_market_open_now", lambda: True)
    ctx = _supervisor_ctx(is_connected=MagicMock(return_value=False), data_age_ms=10)
    fallback = _Fallback(running=False)
    fallback.start.side_effect = RuntimeError("start boom")

    with caplog.at_level(logging.WARNING, logger="nifty_scalper_bot.core.app"):
        await app._polling_failover_supervisor_iteration(
            ctx, fallback, quote_stale_ms=120000, degraded_since=0.0, recovered_since=None, activate_after=0.0
        )

    assert "POLLING_FALLBACK_START_FAILED" in caplog.text
    assert "start boom" in caplog.text
