"""Structural tests confirming the polling-fallback supervisor and the
DATA_WARMUP readiness gate are now market-session aware."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from nifty_scalper_bot.core import app


_APP_PATH = Path(__file__).resolve().parents[2] / "src" / "nifty_scalper_bot" / "core" / "app.py"


def _source() -> str:
    return _APP_PATH.read_text(encoding="utf-8")


def test_polling_fallback_skipped_when_market_closed():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    # The supervisor must check market_open and short-circuit with
    # POLLING_FALLBACK_SKIPPED reason=market_closed.
    assert "polling_fallback_skipped:{spot_symbol}:market_closed" in s
    assert "POLLING_FALLBACK_SKIPPED reason=market_closed" in s
    assert 'market_open = bool(_safe_supervisor_call("is_market_open_now", is_market_open_now, default=False))' in s


def test_within_threshold_polling_skip_log_is_throttled():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    # The within_spot_stale_threshold log must run through log_throttled now.
    assert (
        re.search(
            r"polling_fallback_skipped:\{spot_symbol\}:within_spot_stale_threshold",
            s,
        )
        is not None
    )


def test_polling_fallback_activate_kept_for_open_market():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "POLLING_FALLBACK_ACTIVATE reason=spot_stale" in s


def test_data_warmup_combines_market_closed_and_quote_denied():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "data_warmup_reasons.append(\"broker_quote_access_denied\")" in s
    assert "data_warmup_reasons.append(\"market_closed\")" in s
    assert ',".join(data_warmup_reasons)' in s.replace(' ', '') or \
        '",".join(data_warmup_reasons)' in s


def test_market_session_state_logged_on_startup():
    """Args: none. Returns: None. Raises: AssertionError."""
    s = _source()
    assert "MARKET_SESSION_STATE state=%s" in s


class _Fallback:
    def __init__(self) -> None:
        self.start = MagicMock()
        self.stop = MagicMock()
        self.set_websocket_mode = MagicMock()
        self._running = False

    def is_running(self) -> bool:
        return self._running


def _supervisor_ctx(*, is_connected=True, feed_health=None, data_age_ms=10**9):
    if feed_health is None:
        feed_health = {
            "futures_fresh": False,
            "options_fresh": False,
            "spot_fresh": False,
            "spot_symbol": "NSE:NIFTY",
            "spot_age_ms": 10**9,
        }
    return SimpleNamespace(
        websocket_manager=SimpleNamespace(is_connected=is_connected),
        market_data_manager=SimpleNamespace(
            trading_feed_health=MagicMock(return_value=feed_health),
            data_age_ms=MagicMock(return_value=data_age_ms),
            ensure_spot_reference_fresh=MagicMock(),
        ),
    )


@pytest.mark.asyncio
async def test_polling_failover_supervisor_handles_tuple_is_connected(caplog, monkeypatch):
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
async def test_polling_failover_supervisor_handles_tuple_market_open_fn(caplog, monkeypatch):
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
async def test_polling_failover_supervisor_offmarket_no_activation(monkeypatch):
    monkeypatch.setattr(app, "is_market_open_now", lambda: False)
    ctx = _supervisor_ctx(is_connected=False)
    fallback = _Fallback()

    await app._polling_failover_supervisor_iteration(
        ctx,
        fallback,
        quote_stale_ms=1000,
        degraded_since=0.0,
        recovered_since=None,
        activate_after=0.0,
    )

    fallback.start.assert_not_called()
