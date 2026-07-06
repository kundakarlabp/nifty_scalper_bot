"""Runtime hardening hooks for WebSocket tick ingress."""

from __future__ import annotations

import asyncio
from datetime import datetime
import time
from typing import Any

from nifty_scalper_bot.streaming.websocket_manager import ConnectionState

_INSTALLED_ATTR = "_market_data_hardening_installed"
_ORIGINAL_BUILD_ATTR = "_market_data_hardening_original_build_ticker"
_ORIGINAL_TRADING_WINDOW_ATTR = "_market_data_hardening_original_is_within_trading_window"


def install_websocket_market_data_hardening(manager_cls: type[Any]) -> None:
    """Install idempotent WebSocketManager hardening hooks."""
    if bool(getattr(manager_cls, _INSTALLED_ATTR, False)):
        return

    original_build_ticker = getattr(manager_cls, "_build_ticker", None)
    if callable(original_build_ticker):
        setattr(manager_cls, _ORIGINAL_BUILD_ATTR, original_build_ticker)

        def _build_ticker_with_safe_close(self: Any) -> Any:
            ticker = original_build_ticker(self)
            _wrap_ticker_close(self, ticker)
            return ticker

        setattr(manager_cls, "_build_ticker", _build_ticker_with_safe_close)

    original_window = getattr(manager_cls, "_is_within_trading_window", None)
    if callable(original_window):
        setattr(manager_cls, _ORIGINAL_TRADING_WINDOW_ATTR, original_window)
        setattr(manager_cls, "_is_within_trading_window", _is_within_trading_window_hardened)

    setattr(manager_cls, "_on_ticks", _hardened_on_ticks)
    setattr(manager_cls, _INSTALLED_ATTR, True)


def _wrap_ticker_close(manager: Any, ticker: Any) -> None:
    """Make ticker.close best-effort so cleanup cannot interrupt reconnect."""
    close = getattr(ticker, "close", None)
    if not callable(close) or bool(getattr(ticker, "_nifty_safe_close_installed", False)):
        return

    def _safe_close(*args: Any, **kwargs: Any) -> Any:
        try:
            return close(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            logger = getattr(manager, "_logger", None)
            if logger is not None:
                logger.warning(
                    "WS_TICKER_CLOSE_SUPPRESSED error=%r",
                    exc,
                    exc_info=True,
                    extra={"event": "WS_TICKER_CLOSE_SUPPRESSED", "error": repr(exc)},
                )
            return None

    try:
        setattr(ticker, "close", _safe_close)
        setattr(ticker, "_nifty_safe_close_installed", True)
    except Exception as exc:  # pragma: no cover
        logger = getattr(manager, "_logger", None)
        if logger is not None:
            logger.debug("WS safe-close wrapper skipped: %s", exc)


def _is_within_trading_window_hardened(self: Any) -> bool:
    """Use the configured trading timezone instead of a hard-coded IST object."""
    if not self._trading_window_enabled:
        return True

    now = datetime.now(self._trading_tz)
    if now.weekday() >= 5:
        return False

    now_time = now.time().replace(tzinfo=None)
    allowed = self._trading_start <= now_time <= self._trading_end
    if not allowed:
        self._logger.debug(
            "WS window blocked | now=%s | start=%s | end=%s | tz=%s",
            now_time,
            self._trading_start,
            self._trading_end,
            self._trading_tz,
        )
    return allowed


def _hardened_on_ticks(self: Any, ws: Any, ticks: Any) -> None:
    """Route each WS batch through exactly one ingress path."""
    del ws
    if not ticks:
        return
    if not isinstance(ticks, list):
        self._logger.error("Invalid ticks payload type: %s", type(ticks))
        return

    mdm = getattr(self, "_market_data_manager", None)
    process_ticks = getattr(mdm, "process_ticks", None)
    dispatched_to_mdm = False
    if callable(process_ticks):
        try:
            process_ticks(ticks)
            dispatched_to_mdm = True
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _on_ticks.process_ticks: %s", exc)

    now = time.monotonic()
    self._last_tick_mono = now
    self._last_pong_mono = now
    restored = not self._connected.is_set() or self._state != ConnectionState.CONNECTED
    self._connected.set()
    self._state = ConnectionState.CONNECTED
    self._stream_health = "healthy"
    self._circuit.failures = 0
    self._circuit.open_until_mono = 0.0
    if restored:
        self._logger.info(
            "WEBSOCKET_CONNECTION_RESTORED_BY_TICK",
            extra={"event": "WEBSOCKET_CONNECTION_RESTORED_BY_TICK"},
        )

    update_authoritative = getattr(mdm, "update_authoritative_ticks", None)
    if callable(update_authoritative):
        try:
            update_authoritative(ticks)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _on_ticks.update_authoritative: %s", exc)

    if self._fallback_active and self._fallback_stop_callback is not None:
        self._fallback_active = False
        try:
            self._fallback_stop_callback()
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure in _on_ticks.fallback_stop: %s", exc)

    if not self._first_tick_logged and ticks:
        first = ticks[0]
        first_token = first.get("instrument_token", "?") if isinstance(first, dict) else "?"
        self._first_tick_logged = True
        self._logger.info(
            "FIRST_TICK_RECEIVED instrument_token=%s — pipeline is live",
            first_token,
        )

    if dispatched_to_mdm:
        return

    callback = self._on_tick_callback
    if not callable(callback):
        return

    for tick in ticks:
        if not isinstance(tick, dict) or "instrument_token" not in tick:
            self._logger.debug(
                "Skipping tick without instrument_token: %s",
                list(tick.keys())[:5] if isinstance(tick, dict) else type(tick),
            )
            continue
        try:
            result = callback(tick)
            if asyncio.iscoroutine(result):
                loop = self._resolve_loop()
                self._schedule_coroutine(loop, result)
        except Exception as exc:  # noqa: BLE001
            self._logger.error("Failure dispatching tick: %s", exc, exc_info=True)
